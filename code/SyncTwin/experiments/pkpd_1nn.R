
args <- commandArgs(TRUE)

sim_id = args[1] # sync3
seed = args[2] # 100
set.seed(as.integer(seed))

data_path = paste0('data/', sim_id, '-seed-', seed, '/')

nn1 = function(X1, X0){
    distance = (colSums((X1[, 1] - X0)^2))
    val = (distance == min(distance)) * 1.

    list(solution.w = matrix(val, nrow=ncol(X0), ncol=1))
}

X0 = read.csv(paste0(data_path, 'test-', 'X0', '.csv'), header=FALSE)
X0 = as.matrix(X0)
X0 = X0[1:3, ]

X1 = read.csv(paste0(data_path, 'test-', 'X1', '.csv'), header=FALSE)
X1 = as.matrix(X1)
X1 = X1[1:3, ]

Y_control = read.csv(paste0(data_path, 'test-', 'Y_control', '.csv'), header=FALSE)
Y_control = as.matrix(Y_control)

Y_treated = read.csv(paste0(data_path, 'test-', 'Y_treated', '.csv'), header=FALSE)
Y_treated = as.matrix(Y_treated)

Treatment_effect = read.csv(paste0(data_path, 'test-', 'Treatment_effect', '.csv'), header=FALSE)
Treatment_effect = as.matrix(Treatment_effect)
te_err_mat = matrix(0, nrow=nrow(Treatment_effect), ncol=ncol(Treatment_effect))
w_mat = matrix(0, nrow=ncol(Y_control), ncol=ncol(X1))

for (i in 1:ncol(X1)){
    X1_slice = X1[, i, drop=FALSE]
    res = nn1(X1_slice, X0)

    y_hat = Y_control %*% res$solution.w
    Y_treated_slice = Y_treated[, i, drop=FALSE]
    te_est = Y_treated_slice - y_hat
    te_err = Treatment_effect[, i, drop=FALSE] - te_est
    te_err_mat[, i] = te_err
    w_mat[, i] = res$solution.w
}

write.csv(w_mat, file=paste0(data_path, 'w-nn', '.csv'))
