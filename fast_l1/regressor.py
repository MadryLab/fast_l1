from argparse import ArgumentParser
from threading import Thread

import torch as ch
from tqdm import tqdm
from cupy import ElementwiseKernel

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
    'out = (X_bool - mean) / std',
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
    return inner_products.abs().max(dim=0).values / n

def calc_stats(loader):
    n, X_avg, X_std = 0., 0., 0.
    for X, y, _ in loader:
        X_avg += X.sum(dim=0).float()
        X_std += X.pow(2).sum(dim=0).float()
        n += y.shape[0]
    X_avg /= n
    X_std /= n
    X_std -= X_avg.pow(2)
    X_std.pow_(0.5)
    return X_avg, X_std

def get_num_examples(loader):
    largest_ind, n_ex = 0, 0.
    for bool_X, _, idx in loader:
        n_ex += float(bool_X.shape[0])
        largest_ind = max(largest_ind, idx.max().cpu().item())
    
    return largest_ind, n_ex

def eval_saga(weight, bias, loader, stats):
    _, n_ex = get_num_examples(loader)
    X, y, _ = next(iter(loader))
    batch_size, num_inputs, num_outputs = y.shape[0], X.shape[1], y.shape[1]

    residual = ch.zeros((batch_size, num_outputs), dtype=ch.float32, device=weight.device)
    total_loss = ch.zeros(num_outputs, dtype=ch.float32, device=weight.device)
    X = ch.empty(batch_size, num_inputs, dtype=ch.float32, device=weight.device)
    mm_mu, mm_sig = stats

    iterator = tqdm(loader)
    total_loss[:] = 0.
    for bool_X, y, idx in iterator:
        # Previous residuals
        X.copy_(bool_X)
        normalize(X, mm_mu, mm_sig, X)

        # Compute residuals
        y -= bias
        ch.addmm(input=y, mat1=X, mat2=weight, out=residual, beta=-1)

        residual.pow_(2)
        losses = residual.sum(0)
        total_loss.add_(losses, alpha=0.5)

    return total_loss / n_ex

def tensor_factory(dtype, device):
    def make_tensor(*shape):
        return ch.zeros(shape, dtype=dtype, device=device)
    return make_tensor

def train_saga(weight, bias, loader, val_loader, *, 
               lr, start_lams, lam_decay, end_lams, 
               early_stop_freq=2, early_stop_eps=1e-5):
    largest_ind, n_ex = get_num_examples(loader)
    zeros = tensor_factory(ch.float32, weight.device)
    lam = start_lams.clone()
    X, y, _ = next(iter(loader))
    batch_size, num_inputs, num_outputs = y.shape[0], X.shape[1], y.shape[1]

    a_table = zeros(largest_ind + batch_size, num_outputs).cpu().pin_memory()
    shuttle = zeros(batch_size, num_outputs).cpu().pin_memory()

    w_grad_avg = zeros(*weight.shape)
    w_saga = zeros(*weight.shape)
    b_grad_avg = zeros(*bias.shape)

    residual = zeros(batch_size, num_outputs)
    total_loss, total_loss_prev = zeros(num_outputs), zeros(num_outputs)
    total_loss += float('inf')
    
    best_losses, best_lambdas = zeros(num_outputs), zeros(num_outputs)
    best_losses += float('inf')

    X = zeros(batch_size, num_inputs)
    train_stats = calc_stats(loader)
    mm_mu, mm_sig = train_stats
    t = 0
    while True:
        iterator = tqdm(loader)
        if early_stop_freq and t % early_stop_freq == 0: 
            total_loss_prev.copy_(total_loss)
        total_loss[:] = 0.
        thr = None
        for bool_X, y, idx in iterator:
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
            bias.add_(b_saga, alpha=-lr)

            # update table and averages
            residual += a_prev

            # Move data to the residual while other stuff happens, don't
            # really need it until the next iteration
            if thr is not None: thr.join()
            def do_work(_idx): a_table.index_copy_(0, _idx, shuttle)
            shuttle.copy_(residual, non_blocking=True)
            thr = Thread(target=do_work, args=(idx.cpu(),))
            thr.start()
            
            # Update average gradients
            avg_grad_update(w_grad_avg, w_saga, batch_size, n_ex)
            avg_grad_update(b_grad_avg, b_saga, batch_size, n_ex)

            # Thresholding operation
            fast_threshold(weight, lr * lam)

            residual.pow_(2)
            losses = residual.sum(0)
            total_loss.add_(losses, alpha=0.5)

        w_cpu = weight.cpu()
        total_loss /= n_ex 
        total_loss += lam * ch.norm(weight, p=1, dim=0)

        # Measure progress
        if early_stop_freq and t % early_stop_freq == early_stop_freq - 1:
            done_optimizing = (total_loss >= total_loss_prev - early_stop_eps)
            if val_loader is not None:
                val_losses = eval_saga(weight, bias, val_loader, train_stats)
                # Find indices that (a) we're done with and (b) val loss is better
                val_loss_improved = (val_losses < best_losses)
                val_loss_improved |= (val_losses < 0)
                val_loss_improved &= done_optimizing

                best_losses = ch.where(val_loss_improved, val_losses, best_losses)
                best_lambdas = ch.where(val_loss_improved, lam, best_lambdas)

            lam_decays = ch.where(done_optimizing, lam_decay, 1.)
            lam *= lam_decays
            if ch.all(lam < end_lams): 
                break

        nnz = (ch.abs(w_cpu) > 1e-5).sum(0).float().mean().item()
        total = weight.shape[0]
        print(f"epoch {t} | "
                f"obj {total_loss.cpu().mean().item():.5f} | "
                f"weight nnz {nnz}/{total} ({nnz/total:.4f}) | "
                f"% examples done {(lam < end_lams).float().mean():.2f}")
        t += 1

    return best_lambdas 