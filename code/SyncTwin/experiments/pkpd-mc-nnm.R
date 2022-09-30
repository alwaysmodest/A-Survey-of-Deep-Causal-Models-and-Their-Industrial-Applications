library(softImpute)

args <- commandArgs(TRUE)

sim_id = args[1] # sync3
seed = args[2] # 100
set.seed(as.integer(seed))

data_path = paste0('data/', sim_id, '-seed-', seed, '/')


X0 = read.csv(paste0(data_path, 'test-', 'X0', '.csv'), header=FALSE)
X0 = as.matrix(X0)

X1 = read.csv(paste0(data_path, 'test-', 'X1', '.csv'), header=FALSE)
X1 = as.matrix(X1)

Y_control = read.csv(paste0(data_path, 'test-', 'Y_control', '.csv'), header=FALSE)
Y_control = as.matrix(Y_control)

Y_treated = read.csv(paste0(data_path, 'test-', 'Y_treated', '.csv'), header=FALSE)
Y_treated = as.matrix(Y_treated)

Treatment_effect = read.csv(paste0(data_path, 'test-', 'Treatment_effect', '.csv'), header=FALSE)
Treatment_effect = as.matrix(Treatment_effect)

X0_v = read.csv(paste0(data_path, 'val-', 'X0', '.csv'), header=FALSE)
X0_v = as.matrix(X0_v)

Y_control_v = read.csv(paste0(data_path, 'val-', 'Y_control', '.csv'), header=FALSE)
Y_control_v = as.matrix(Y_control_v)
te_val = function(X0, Y_control, lambda=1, multi_dim=TRUE){
    dimx = nrow(X0) / 3
    if (multi_dim){
        mat_0 = rbind(Y_control, X0)
    } else {
        dimx = nrow(X0) / 3
        mat_0 = rbind(Y_control, X0[ (2*dimx+1):(3*dimx), ])
    }
    np = nrow(mat_0) * ncol(mat_0)
    ix=seq(np)
    missfrac = 0.3
    imiss=sample(ix,np*missfrac,replace=FALSE)
    xna=mat_0
    xna[imiss]=NA
    res = softImpute(xna, rank.max = min(dim(xna))-1 , lambda = lambda, type = "svd")
    xna_hat=complete(xna, res)
    mean(abs(xna_hat[imiss] - mat_0[imiss]))
}
# search_list = c(0.01, 0.1, 1, 2, 3, 10)
search_list = c(3, 4, 5, 8, 10)
err = list()

for (l in search_list){
    e = te_val(X0_v, Y_control_v, lambda=l)
    err[length(err) + 1] = e
}
best_ind = which.min(unlist(err))
lambda = search_list[best_ind]
te_impute = function(X0, X1, Y_control, Y_treated, lambda=1, multi_dim=TRUE){
    dimx = nrow(X0) / 3
    if (multi_dim){
        mat_0 = rbind(Y_control, X0)
        mat_1 = rbind(matrix(NA, nrow=nrow(Y_treated), ncol=ncol(Y_treated)), X1)
    } else {
        dimx = nrow(X0) / 3
        mat_0 = rbind(Y_control, X0[ (2*dimx+1):(3*dimx), ])
        mat_1 = rbind(matrix(NA, nrow=nrow(Y_treated), ncol=ncol(Y_treated)), X1[ (2*dimx+1):(3*dimx), ])
    }
    mat_total = cbind(mat_1, mat_0)
    res = softImpute(mat_total, rank.max = min(dim(mat_total))-1 , lambda = lambda, type = "svd")
    mat_total_hat=complete(mat_total, res)
    y_est = mat_total_hat[1:nrow(Y_treated), 1:ncol(Y_treated)]
    te = Y_treated - y_est
    list(te, res)
}
ret = te_impute(X0, X1, Y_control, Y_treated, lambda)
te = ret[[1]]
res = ret[[2]]
mae = mean(abs(te - Treatment_effect))
mae_sd = sd(abs(te - Treatment_effect))  / sqrt(ncol(Y_treated))
print(mae)
print(mae_sd)