from threading import Thread
from typing import Optional

import numpy as np
import torch as ch
from tqdm import tqdm
from cupy import ElementwiseKernel

from fast_l1.logger import Logger

kernel = ElementwiseKernel(
    'float32 data, float32 lamb',
    'float32 out',
    'out = (data - lamb) * (data > lamb) + (data + lamb) * (data < -lamb)',
    'soft_thresholding'
)


def fast_threshold(data, lamb):
    kernel(data, lamb, data)


mix_grad_kernel = ElementwiseKernel(
    'float32 grad_avg, float32 grad_saga, float32 B, float32 n_ex',
    'float32 out',
    'out = (1 - B / n_ex) * grad_avg + (B / n_ex) * grad_saga',
    'grad_update'
)


def avg_grad_update(grad_avg, grad_saga, B, n_ex):
    mix_grad_kernel(grad_avg, grad_saga, B, n_ex, grad_avg)


normalize_kernel = ElementwiseKernel(
    'float32 X_bool, float32 mean, float32 std',
    'float32 out',
    'out = (X_bool - mean) / (std + 1e-32)',
    'ez_normalize'
)


def normalize(X_bool, mean, std, X):
    normalize_kernel(X_bool, mean, std, X)


# Calculate maximum regularization
def calc_max_lambda(loader):
    n, y_sum = 0., 0.
    # calculate mean
    for X, y, _ in loader:
        y_sum += y.sum(dim=0).float()
        n += y.shape[0]
    y_bar = y_sum / n

    # calculate maximum regularization
    inner_products = 0
    for X, y, _ in loader:
        y_map = (y - y_bar)
        inner_products += X.T.float().mm(y_map)
    return ch.abs(inner_products).max(dim=0).values / n


def calc_stats(loader):
    n, X_avg, X_std = 0., 0., 0.
    for X, y, _ in loader:
        X_avg += X.sum(dim=0).float()
        X_std += X.pow(2).sum(dim=0).float()
        n += y.shape[0]
    X_avg /= n
    X_std /= n
    X_std -= ch.pow(X_avg, 2)
    X_std.pow_(0.5)
    return X_avg, X_std


def get_num_examples(loader):
    largest_ind, n_ex = 0, 0.
    for bool_X, _, idx in loader:
        n_ex += float(bool_X.shape[0])
        largest_ind = max(largest_ind, idx.max().cpu().item())

    print('Largest index', largest_ind)
    return largest_ind, n_ex


def eval_saga(weight, bias, loader, stats,
              batch_size, num_inputs, num_outputs,
              index_mapping):
    residual = ch.zeros((batch_size, num_outputs),
                        dtype=ch.float32, device=weight.device)
    total_loss = ch.zeros(num_outputs,
                          dtype=ch.float32, device=weight.device)
    X = ch.empty(batch_size, num_inputs,
                 dtype=ch.float32, device=weight.device)
    mm_mu, mm_sig = stats
    y_buf = ch.empty(batch_size, num_outputs,
                     dtype=ch.float32, device=weight.device)

    iterator = tqdm(loader)
    total_loss[:] = 0.
    n_ex = 0

    for bool_X, y, idx in iterator:
        # Previous residuals
        n_ex += bool_X.shape[0]
        X.copy_(bool_X)
        normalize(X, mm_mu, mm_sig, X)

        # Compute residuals
        y_buf[:] = y[:, index_mapping]
        y_buf -= bias
        ch.addmm(input=y_buf, mat1=X, mat2=weight, out=residual, beta=-1)

        residual.pow_(2)
        losses = residual.sum(0)
        total_loss.add_(losses)

    return total_loss / n_ex


def tensor_factory(dtype, device):
    def make_tensor(*shape):
        return ch.zeros(shape, dtype=dtype, device=device)
    return make_tensor


def swap_inds(tens, inds1, inds2, dim=1):
    if dim == 0:
        tens[ch.cat([inds1, inds2])] = tens[ch.cat([inds2, inds1])]
    elif dim == 1:
        tens[:, ch.cat([inds1, inds2])] = tens[:, ch.cat([inds2, inds1])]
    else:
        raise ValueError("invalid dim value")


def train_saga(weight, bias, loader, val_loader, *,
               lr, start_lams, lam_decay, num_lambdas,
               early_stop_freq=2, early_stop_eps=1e-5,
               logdir: Optional[str] = None,
               update_bias=True,
               min_lams_to_try=10):
    largest_ind, n_ex = get_num_examples(loader)
    zeros = tensor_factory(ch.float32, weight.device)
    bool_zeros = tensor_factory(ch.bool, weight.device)

    X, y, _ = next(iter(loader))
    batch_size, num_inputs, num_outputs = y.shape[0], X.shape[1], y.shape[1]

    logger = None
    if logdir is not None:
        logger = Logger(logdir, fields={
            'train_mse': np.float32,
            'val_mse': np.float32,
            'deltas': np.float32,
            'lambda': np.float32,
            'best_lambda': np.float32,
            'best_mse': np.float32,
            'weight_norm': np.float32,
            'weight_nz': np.float32,
            'done_optimizing_inner': np.bool_,
            'still_optimizing_outer': np.bool_
        }, field_size=num_outputs, cnk_size=10_000)

    a_table = zeros(largest_ind + batch_size, num_outputs).cpu().pin_memory()
    shuttle = zeros(batch_size, num_outputs).cpu().pin_memory()

    # Stateful variables
    # lam = start_lams.clone().to(weight.device)
    start_lams = start_lams.to(weight.device)
    num_lambdas += 1
    lam = zeros(*bias.shape)
    w_grad_avg = zeros(*weight.shape)
    b_grad_avg = zeros(*bias.shape)

    done_opt_inner = bool_zeros(num_outputs)
    still_opt_outer = ~bool_zeros(num_outputs)
    got_worse = bool_zeros(num_outputs)

    last_mse = zeros(num_outputs) + ch.inf
    best_mse = zeros(num_outputs) + ch.inf
    lambdas_done = ch.zeros(num_outputs, dtype=ch.long, device=weight.device)
    best_lambdas = zeros(num_outputs)

    # Pre-allocated loop variables
    w_saga = zeros(*weight.shape)
    b_saga = zeros(*bias.shape)
    residual = zeros(batch_size, num_outputs)

    prev_w = ch.zeros_like(weight)
    deltas = zeros(num_outputs)
    new_fin_mask = bool_zeros(num_outputs)

    # Preallocated inputs
    X = zeros(batch_size, num_inputs)
    train_stats = calc_stats(loader)
    mm_mu, mm_sig = train_stats
    t = 0

    # This is to keep track of early stopping
    # Garbage variable
    deltas_inds = ch.zeros(num_outputs, dtype=ch.long, device=weight.device)

    # This is for logging
    train_losses = zeros(num_outputs)
    total_train_losses = zeros(num_outputs)

    # These are buffers for fastmm
    y_buf = zeros(batch_size, num_outputs)

    a_prev = zeros(batch_size, num_outputs)
    index_mapping = ch.arange(num_outputs).cuda()
    num_keep = num_outputs
    try:
        while True:
            iterator = tqdm(loader)
            thr = None
            prev_w[:] = weight
            # Try rearranging weight vector here
            for bool_X, y, idx in iterator:
                a_prev[:, :num_keep].copy_(a_table[idx.cpu(), :num_keep],
                                           non_blocking=True)

                X.copy_(bool_X)
                normalize(X, mm_mu, mm_sig, X)

                # Compute residuals
                y_buf[:, :num_keep] = y[:, index_mapping[:num_keep]]
                y_buf[:, :num_keep] -= bias[:num_keep]

                ch.addmm(input=y_buf[:, :num_keep], mat1=X,
                         mat2=weight[:, :num_keep],
                         out=residual[:, :num_keep], beta=-1)

                residual -= a_prev

                ch.mm(X.T, residual[:, :num_keep], out=w_saga[:, :num_keep])

                w_saga /= batch_size
                w_saga += w_grad_avg

                ch.sum(residual[:, :num_keep], dim=0, out=b_saga[:num_keep])
                b_saga /= batch_size
                b_saga += b_grad_avg

                # Gradient steps for weight
                weight[:, :num_keep].add_(w_saga[:, :num_keep], alpha=-lr)
                if update_bias:
                    bias[:num_keep].add_(b_saga[:num_keep], alpha=-lr)

                # update table and averages
                residual += a_prev

                # Move data to the residual while other stuff happens, don't
                # really need it until the next iteration
                if thr is not None:
                    thr.join()

                def do_work(_idx):
                    a_table[:, :num_keep].index_copy_(0, _idx,
                                                      shuttle[:, :num_keep])

                shuttle[:, :num_keep].copy_(residual[:, :num_keep],
                                            non_blocking=True)
                thr = Thread(target=do_work, args=(idx.cpu(),))
                thr.start()

                # Update average gradients
                avg_grad_update(w_grad_avg[:, :num_keep], w_saga[:, :num_keep],
                                batch_size, n_ex)
                avg_grad_update(b_grad_avg[:num_keep], b_saga[:num_keep],
                                batch_size, n_ex)

                # Thresholding operation
                fast_threshold(weight[:, :num_keep], lr * lam[:num_keep])

                residual[:, :num_keep].pow_(2)
                ch.sum(residual[:, :num_keep], dim=0,
                       out=train_losses[:num_keep])
                total_train_losses[:num_keep] += train_losses[:num_keep]

            # https://glmnet.stanford.edu/articles/glmnet.html#appendix-0-convergence-criteria-1
            prev_w -= weight
            prev_w.pow_(2)
            prev_w *= mm_sig.pow(2)[:, None]
            ch.max(prev_w, dim=0, out=(deltas, deltas_inds))
            ch.lt(deltas, early_stop_eps, out=done_opt_inner)

            data_to_log = {
                'train_mse': total_train_losses / n_ex,
                'val_mse': last_mse,
                'deltas': deltas,
                'lambda': lam,
                'best_lambda': best_lambdas,
                'best_mse': best_mse,
                'weight_norm': weight.norm(dim=0, p=1).cpu(),
                'weight_nz': weight.norm(dim=0, p=0).cpu(),
                'done_optimizing_inner': done_opt_inner,
                'still_optimizing_outer': still_opt_outer
            }
            if logger is not None:
                for name, value in data_to_log.items():
                    logger.log(name, value.cpu().numpy())
                logger.log_index_mapping(index_mapping.cpu().numpy())

            # Decrement lambdas for the ones done optimizing
            if t % early_stop_freq == early_stop_freq - 1:
                lambdas_done += (done_opt_inner & still_opt_outer)
                ch.eq(lambdas_done, num_lambdas, out=new_fin_mask)

                # New value of the MSE
                new_mse = None
                if val_loader:
                    new_mse = eval_saga(weight, bias, val_loader,
                                        train_stats, batch_size,
                                        num_inputs, num_outputs,
                                        index_mapping)

                    # Of the indices done optimizing, see if val loss got worse
                    ch.gt(new_mse, last_mse, out=got_worse)
                    got_worse &= (lambdas_done >= min_lams_to_try)
                    got_worse &= (weight.norm(dim=0, p=0) > 1)

                    best_lambdas = ch.where((new_mse <= best_mse) & done_opt_inner, 
                                             lam, best_lambdas)
                    best_mse = ch.where((new_mse <= best_mse) & done_opt_inner, 
                                         new_mse, best_mse)
                else:
                    got_worse[:] = False

                got_worse &= done_opt_inner
                new_fin_mask |= got_worse
                new_fin_mask &= still_opt_outer
                still_opt_outer[new_fin_mask] = False

                # Where (in the original indices) we are newly done
                new_fin_inds = ch.where(new_fin_mask)[0].cpu()
                new_done = len(new_fin_inds)
                inds_to_swap = ch.arange(num_keep - new_done, num_keep)

                inds_to_swap = inds_to_swap[still_opt_outer[inds_to_swap].cpu()]
                new_fin_inds = new_fin_inds[new_fin_inds < num_keep - new_done]

                for tens in [a_table, w_grad_avg, weight]:
                    swap_inds(tens, inds_to_swap, new_fin_inds)

                to_swap = [bias, lam, b_grad_avg, still_opt_outer, index_mapping,
                           lambdas_done, last_mse, best_lambdas, best_mse]
                if new_mse is not None:
                    to_swap.append(new_mse)

                for tens in to_swap:
                    swap_inds(tens, inds_to_swap, new_fin_inds, 0)

                num_keep = still_opt_outer.sum().cpu().item()

                # Wherever we are done, update the val mse and lambda
                if new_mse is not None:
                    last_mse[done_opt_inner] = new_mse[done_opt_inner]
                else:
                    last_mse[done_opt_inner] = 0.
                done_opt_inner &= still_opt_outer
                lam[done_opt_inner & (lambdas_done == 1)] = \
                    start_lams[done_opt_inner & (lambdas_done == 1)]
                lam[done_opt_inner & (lambdas_done > 1)] *= lam_decay
                done_opt_inner[:] = False

            total_train_losses[:] = 0.
            if ch.all(~still_opt_outer):
                break

            nnz = weight.nonzero().shape[0]
            total = weight.shape[0]
            avg_lambdas_done = lambdas_done.float().mean()
            sparsity = nnz/(weight.shape[1] * total)
            print(f"epoch: {t} | "
                  f"delta: {deltas[still_opt_outer].mean()} | "
                  f"weight nnz {nnz}/{total} ({sparsity:.4f}) | "
                  f"{avg_lambdas_done:.2f} lambdas done on average | "
                  f"{num_keep} examples left | ")
            t += 1
    except KeyboardInterrupt:
        if logger is not None:
            logger.flush()
        print('Interrupted, quitting...')

    index_unmapping = ch.argsort(index_mapping)
    weight[:] = weight[:, index_unmapping]
    bias[:] = bias[index_unmapping]
    lam[:] = lam[index_unmapping]
    best_lambdas[:] = best_lambdas[index_unmapping]
    return best_lambdas
