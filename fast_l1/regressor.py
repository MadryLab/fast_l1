from threading import Thread

import numpy as np
import torch as ch
from tqdm import tqdm
from cupy import ElementwiseKernel
from cox.store import Store, PICKLE

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

    return largest_ind, n_ex


def eval_saga(weight, bias, loader, stats,
              batch_size, num_inputs, num_outputs):
    residual = ch.zeros((batch_size, num_outputs),
                        dtype=ch.float32, device=weight.device)
    total_loss = ch.zeros(num_outputs,
                          dtype=ch.float32, device=weight.device)
    X = ch.empty(batch_size, num_inputs,
                 dtype=ch.float32, device=weight.device)
    mm_mu, mm_sig = stats

    iterator = tqdm(loader)
    total_loss[:] = 0.
    n_ex = 0
    for bool_X, y, idx in iterator:
        # Previous residuals
        n_ex += bool_X.shape[0]
        X.copy_(bool_X)
        normalize(X, mm_mu, mm_sig, X)

        # Compute residuals
        y -= bias
        ch.addmm(input=y, mat1=X, mat2=weight, out=residual, beta=-1)

        residual.pow_(2)
        losses = residual.sum(0)
        total_loss.add_(losses)

    return total_loss / n_ex


def tensor_factory(dtype, device):
    def make_tensor(*shape):
        return ch.zeros(shape, dtype=dtype, device=device)
    return make_tensor


def train_saga(weight, bias, loader, val_loader, *,
               lr, start_lams, lam_decay, num_lambdas,
               early_stop_freq=2, early_stop_eps=1e-5,
               cox_store: Store = None,
               logdir: str = None,
               update_bias=True,
               dynamic_resize=True):
    largest_ind, n_ex = get_num_examples(loader)
    zeros = tensor_factory(ch.float32, weight.device)
    bool_zeros = tensor_factory(ch.bool, weight.device)

    lam = start_lams.clone().to(weight.device)
    X, y, _ = next(iter(loader))
    batch_size, num_inputs, num_outputs = y.shape[0], X.shape[1], y.shape[1]

    logger = None
    if logdir is not None:
        logger = Logger(logdir, fields={
            'train_mse': (np.float32, num_outputs),
            'val_mse': (np.float32, num_outputs),
            'lambda': (np.float32, num_outputs),
            'weight_norm': (np.float32, num_outputs),
            'done_optimizing_inner': (np.bool_, num_outputs),
            'still_optimizing_outer': (np.bool_, num_outputs)
        }, cnk_size=100_000)

    a_table = zeros(largest_ind + batch_size, num_outputs).cpu().pin_memory()
    shuttle = zeros(batch_size, num_outputs).cpu().pin_memory()

    w_grad_avg = zeros(*weight.shape)
    w_saga = zeros(*weight.shape)
    b_grad_avg = zeros(*bias.shape)

    residual = zeros(batch_size, num_outputs)

    # w_norm = zeros(num_outputs)
    done_opt_inner = bool_zeros(num_outputs)
    still_opt_outer = ~bool_zeros(num_outputs)
    last_resizer = ch.arange(num_outputs)
    got_worse = bool_zeros(num_outputs)

    X = zeros(batch_size, num_inputs)
    train_stats = calc_stats(loader)
    mm_mu, mm_sig = train_stats
    t = 0

    # This is to keep track of early stopping
    prev_w = ch.zeros_like(weight)
    deltas = zeros(num_outputs)
    deltas_inds = ch.zeros(num_outputs, dtype=ch.long, device=weight.device)
    last_mse = zeros(num_outputs) + ch.inf
    lambdas_done = ch.zeros(num_outputs, dtype=ch.long, device=weight.device)

    # This is for logging
    train_losses = zeros(num_outputs)
    total_train_losses = zeros(num_outputs)

    w_cpu = ch.zeros_like(weight, device=ch.device('cpu'))
    b_cpu = ch.zeros_like(bias, device=ch.device('cpu'))
    try:
        while True:
            iterator = tqdm(loader)
            thr = None
            prev_w[:] = weight
            for bool_X, y, idx in iterator:
                y = y[:, last_resizer]
                # Previous residuals
                a_prev = a_table[idx].cuda(non_blocking=True)
                X.copy_(bool_X)
                normalize(X, mm_mu, mm_sig, X)

                # Compute residuals
                y -= bias
                ch.addmm(input=y, mat1=X, mat2=weight, out=residual, beta=-1)

                residual -= a_prev
                ch.mm(X.T, residual, out=w_saga)

                w_saga /= batch_size
                w_saga += w_grad_avg
                b_saga = residual.sum(0) / batch_size
                b_saga += b_grad_avg

                # Gradient steps for weight
                weight.add_(w_saga, alpha=-lr)
                if update_bias:
                    bias.add_(b_saga, alpha=-lr)

                # update table and averages
                residual += a_prev

                # Move data to the residual while other stuff happens, don't
                # really need it until the next iteration
                if thr is not None:
                    thr.join()

                def do_work(_idx):
                    a_table.index_copy_(0, _idx, shuttle)

                shuttle.copy_(residual, non_blocking=True)
                thr = Thread(target=do_work, args=(idx.cpu(),))
                thr.start()

                # Update average gradients
                avg_grad_update(w_grad_avg, w_saga, batch_size, n_ex)
                avg_grad_update(b_grad_avg, b_saga, batch_size, n_ex)

                # Thresholding operation
                fast_threshold(weight, lr * lam)

                residual.pow_(2)
                ch.sum(residual, dim=0, out=train_losses)
                total_train_losses += train_losses

            # https://glmnet.stanford.edu/articles/glmnet.html#appendix-0-convergence-criteria-1
            prev_w -= weight
            prev_w.pow_(2)
            prev_w *= mm_sig.pow(2)[:, None]
            ch.max(prev_w, dim=0, out=(deltas, deltas_inds))
            ch.lt(deltas, early_stop_eps, out=done_opt_inner)

            data_to_log = {
                'train_mse': total_train_losses / n_ex,
                'val_mse': last_mse,
                'lambda': lam,
                'done_optimizing_inner': done_opt_inner,
                'still_optimizing_outer': still_opt_outer
            }
            if logger is not None:
                for name, value in data_to_log.items():
                    logger.log(name, value.cpu().numpy())

            # Decrement lambdas for the ones done optimizing
            if t % early_stop_freq == early_stop_freq - 1:
                """
                    if done_optimizing_inner[output_ind]:
                        cox_store['weights'].append_row({
                            'index': output_ind,
                            'lambda_ind': lambdas_done[output_ind],
                            'lambda': lam[output_ind],
                            'weight': weight[:, output_ind],
                            'bias': bias[output_ind]
                        })
                """

                lambdas_done += (done_opt_inner & still_opt_outer)
                still_opt_outer[lambdas_done == num_lambdas] = False

                # New value of the MSE
                new_mse = eval_saga(weight, bias, val_loader,
                                    train_stats, batch_size,
                                    num_inputs, num_outputs)

                # Of the indices done optimizing, see if val loss got worse
                got_worse[:] = (new_mse > last_mse) & done_opt_inner

                # Wherever it got worse, stop optimizing and decrement lambda
                lam[got_worse & still_opt_outer] /= lam_decay
                still_opt_outer[got_worse] = False

                # Wherever we are done, update the val mse and lambda
                last_mse[done_opt_inner] = new_mse[done_opt_inner]
                lam[done_opt_inner & still_opt_outer] *= lam_decay
                done_opt_inner[:] = False

            total_train_losses[:] = 0.
            if ch.all(~still_opt_outer):
                break
            elif dynamic_resize and ch.mean(still_opt_outer.float()) < 0.9:
                print('Resizing stuff...')
                w_cpu[:, last_resizer] = weight[:, still_opt_outer]
                b_cpu[:, last_resizer] = bias[still_opt_outer]

                last_resizer = last_resizer[still_opt_outer]
                a_table = a_table[:, still_opt_outer]
                shuttle = shuttle[:, still_opt_outer]
                w_grad_avg = w_grad_avg[:, still_opt_outer]
                w_saga = w_saga[:, still_opt_outer]
                b_grad_avg = b_grad_avg[still_opt_outer]
                done_opt_inner = done_opt_inner[still_opt_outer]
                still_opt_outer = still_opt_outer[still_opt_outer]
                weight = weight[:, still_opt_outer]
                bias = bias[still_opt_outer]

            nnz = weight.nonzero().shape[0]
            total = weight.shape[0]
            print(f"epoch: {t} | delta: {deltas.mean()} | "
                  f"weight nnz {nnz}/{total} ({nnz/(weight.shape[1] * total):.4f}) | "
                  f"{lambdas_done.float().mean():.2f} lambdas done on average")
            t += 1
    except KeyboardInterrupt:
        cox_store.close()
        print('Interrupted, quitting...')
