## Implementations of the functions by
## Zhu, C., & Müller, H. G. (2023). Autoregressive optimal transport models. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(3), 1012-1033.
## https://academic.oup.com/jrsssb/article/85/3/1012/7160826

## Columns of Q are quantile functions supported on the same grid.
## p is the order of autoregressive model
## eta is the learning rate for gradient descent
## Method= "FromMean" or "FromDifference"


ATM <- function(Q, p, eta, Method){
  
  # d is the length of qSup and the number of rows of Q
  d <- dim(Q)[1]
  if(Method=="FromMean"){
    ref <- rowMeans(Q)
    Tf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(ref, Q[,i], method = "natural")})
    ITf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(Q[,i], ref, method = "natural")})
    Grid <- matrix(rep(ref, dim(Q)[2]), d, dim(Q)[2])
  }else if(Method=="FromDifference"){
    Tf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i], Q[,i+1], method = "natural")})
    ITf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i+1], Q[,i], method = "natural")})
    Grid <- Q
  }else{
    stop("invalid method!")
  }
  #n is the number of transport maps constructed
  n <- length(Tf)
  
  ################################################### Initialization of alpha #############################################
  alpha <- rep(0,p)
  X_plus <- sapply(c(1:(n-1)), function(i){return(Tf[[i]](Grid[,i+1], 0))}) - Grid[,c(2:n)]
  Y <- sapply(c(2:(n)), function(i){return(Tf[[i]](Grid[,i], 0))}) - Grid[,c(2:n)]
  
  s1 <- sapply(c(2:n), function(i){
    return(trapzRcpp(Grid[,i], X_plus[,i-1]*X_plus[,i-1])/(Grid[d,i]-Grid[1,i]))
  })
  s2 <- sapply(c(2:n), function(i){
    return(trapzRcpp(Grid[,i], X_plus[,i-1]*Y[,i-1])/(Grid[d,i]-Grid[1,i]))
  })
  alpha_plus <- sum(s2)/sum(s1)
  if(alpha_plus<0) {alpha_plus =0}
 
  L_plus <- mean(
    sapply(c(2:n), function(i){
      return(trapzRcpp(Grid[,i], (Y[,i-1]-alpha_plus*X_plus[,i-1])^2)/(Grid[d,i]-Grid[1,i]))
    })
  )
  
  X_minus <- Grid[,c(2:n)] - sapply(c(1:(n-1)), function(i){return(ITf[[i]](Grid[,i+1], 0))}) 
  s1 <- sapply(c(2:n), function(i){
    return(trapzRcpp(Grid[,i], X_minus[,i-1]*X_minus[,i-1])/(Grid[d,i]-Grid[1,i]))
  })
  s2 <- sapply(c(2:n), function(i){
    return(trapzRcpp(Grid[,i], X_minus[,i-1]*Y[,i-1])/(Grid[d,i]-Grid[1,i]))
  })
  alpha_minus <- sum(s2)/sum(s1)
  if(alpha_minus>0){alpha_minus = 0}
  #if(alpha_minus<(-1)){alpha_minus = -1}
  #L_minus <- sum((Y - alpha_minus*X_minus)^2)
  L_minus <- mean(
    sapply(c(2:n), function(i){
      return(trapzRcpp(Grid[,i], (Y[,i-1]-alpha_minus*X_minus[,i-1])^2)/(Grid[d,i]-Grid[1,i]))
    })
  )
  
  if(L_plus <= L_minus){ 
    loss <- L_plus
    alpha[1] = alpha_plus
  }else{ 
    loss <- L_minus
    alpha[1] = alpha_minus
  }
  
  ##################################################### Gradient descent algorithm ######################################  
  R <- vector("list", length = p)
  if(p>1){
    D <- vector("list", length = p)
    D[[1]] = matrix(rep(1, (n-p)*d), d, n-p)
    lossOld <- Inf
    loss <- 100
    ite <- 0
    while(ite <= 200){
      ####################################### forward pass ##################################################
      for(k in c(1:p)){
        if(alpha[p+1-k] >= 0){
          TFun <- Tf
        }else{
          TFun <- ITf
        }
        
        a <- floor(abs(alpha[p+1-k]))
        r <- abs(alpha[p+1-k]) - a
        R[[k]] =  vector("list", length=(a+1))
        if(k==1){
          X <- Grid[,(p+1):n]
        }else{
          X <- R[[k-1]][[length(R[[k-1]])]]
        }
        b=1
        while(b<=a){
          R[[k]][[b]] <- sapply(c(k:(n+k-p-1)), function(i){return(TFun[[i]](X[,(i+1-k)],0))})
          X <- R[[k]][[b]]
          b=(b+1)
        }
        R[[k]][[b]] <- sapply(c(k:(n+k-p-1)), function(i){
          return(X[,(i+1-k)] + r*(TFun[[i]](X[,(i+1-k)],0) - X[,(i+1-k)]))
        })
      }
      L <- 2*(sapply(c((p+1):n), function(i){return(Tf[[i]](Grid[,i], 0))}) - R[[p]][[length(R[[p]])]])
      #L <- t(weight[c((p+1):n)]*t(L))
      lossOld <- loss
      #loss <- sum(L^2)/4
      loss <- mean(
        sapply(c((p+1):n), function(i){
          return(trapzRcpp(Grid[,i], (L[,i-p])^2/4)/(Grid[d,i]-Grid[1,i]))
        })
      )
      #print(loss)
      if(abs(lossOld - loss)<10^(-8)){break}
      
      ############################################## backward pass ##################################################
      gradient <- rep(0,p)
      for(k in c(1:p)){
        if(alpha[k] > 0){
          TFun <- Tf;  
          sig= matrix(rep(1,(n-p)*d), d, (n-p))
        }else if(alpha[k] < 0){
          TFun <- ITf; 
          sig= matrix(rep(0,(n-p)*d), d, (n-p))
        }else{
          #set.seed(k)
          TFun <- Tf; 
          sig=matrix(runif((n-p)*d), d, (n-p))
          #sig=matrix(rep(-1, (n-p)*d), d, (n-p))
        }
        a <- floor(abs(alpha[k]))
        r <- abs(alpha[k]) - a
        
        if(k<p){
          X <- R[[p-k]][[length(R[[p-k]])]]
        }else{
          X <- Grid[,c((p+1):n)]
        }
        
        B <- matrix(rep(1,(n-p)*d), d, (n-p))
        if(a>1){
          B <- sapply(c((p+1-k):(n-k)), function(i){return(TFun[[i]](X[,(i-p+k)],1))})
          b=2
          while(b<=a){
            B <- sapply(c((p+1-k):(n-k)), function(i){return(TFun[[i]](R[[p+1-k]][[b-1]][,(i-p+k)],1))})*B
            b = (b+1)
          }
          X <- R[[p+1-k]][[a]]
        }
        
        D[[k+1]] <- sapply(c((p+1-k):(n-k)), function(i){
          return(1 + r*(TFun[[i]](X[,(i-p+k)],1)-1))
        })*B
        
        A <- sapply(c((p+1-k):(n-k)), function(i){
          v <- X[,(i-p+k)]
          sigj <- sig[,(i-p+k)]
          return(sigj*(Tf[[i]](v,0) - v) + (1-sigj)*(v-ITf[[i]](v,0)))
        })
        #gradient[k] <- sum(L*D[[k]]*A)
        gradient[k] <- sum(
          sapply(c((p+1):n), function(i){
            return(trapzRcpp(Grid[,i], L[,i-p]*D[[k]][,i-p]*A[,i-p])/(Grid[d,i]-Grid[1,i]))
          })
        )
      }
      if(sum(gradient^2)>1){
        gradient <- gradient/sqrt(sum(gradient^2))
      }
      alpha <- alpha + eta*gradient
      ite = ite +1
    }
  }
  
  ############################################ make predictions ########################################## 
  R <- vector("list", length = p)
  for(k in c(1:p)){
    if(alpha[p+1-k] >= 0){
      TFun <- Tf
    }else{
      TFun <- ITf
    }
    
    a <- floor(abs(alpha[p+1-k]))
    r <- abs(alpha[p+1-k]) - a
    
    if(k==1){
      v <- Grid[,dim(Grid)[2]]
    }else{
      v <- R[[k-1]][[length(R[[k-1]])]]
    }
    b=1
    while(b<=a){
      R[[k]][[b]] <- TFun[[n-p+k]](v,0)
      v <- R[[k]][[b]]
      b=(b+1)
    }
    R[[k]][[b]] <- v + r*(TFun[[n+k-p]](v,0) - v)
  }
  pred <- R[[p]][[length(R[[p]])]]
  return(list("alpha" = alpha, "pred"=pred, "TrainingLoss" = loss, "L_plus"=L_plus, "L_minus"=L_minus ))
}

cv.ATM <- function(Q1, Q2, qSup, eta, Method){
  n1 <- dim(Q1)[2]
  n2 <- dim(Q2)[2]
  Q <- cbind(Q1, Q2)
  
  #pList <- c(1, 2, 3, 5, 7)
  pList <- c(1, 2)
  
  loss <- rep(0, length(pList))
  for (k in c(1:length(pList))){
    p <- pList[k]
    for (i in c(1:n1)) {
      Qtemp <- Q[,c(i:(i+n2-1))]
      target <- Q[,c(i+n2)] 
      atm <- ATM(Qtemp, p, eta, Method)
      loss[k] <- loss[k] + sqrt(trapzRcpp(qSup, (atm$pred - target)^2))
    }
    loss[k] <- loss[k]/n1
  }
  return(pList[which.min(loss)])
}

predict.atm <- function(alpha, Q, method, predictor.index){
  p <- length(alpha)
  d <- dim(Q)[1]
  if(method=="FromMean"){
    ref <- rowMeans(Q)
    Tf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(ref, Q[,i], method = "natural")})
    ITf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(Q[,i], ref, method = "natural")})
    grid <- ref
  }else if(method=="FromDifference"){
    Tf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i], Q[,i+1], method = "natural")})
    ITf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i+1], Q[,i], method = "natural")})
    grid <- Q[,  predictor.index+1]
  }else{
    stop("invalid method!")
  }
  
  R <- vector("list", length = p)
  for(k in c(1:p)){
    if(alpha[p+1-k] >= 0){
      TFun <- Tf[c((predictor.index-p+1):predictor.index)]
    }else{
      TFun <- ITf[c((predictor.index-p+1):predictor.index)]
    }
    
    a <- floor(abs(alpha[p+1-k]))
    r <- abs(alpha[p+1-k]) - a
    
    if(k==1){
      v <- grid
    }else{
      v <- R[[k-1]][[length(R[[k-1]])]]
    }
    b=1
    while(b<=a){
      R[[k]][[b]] <- TFun[[k]](v,0)
      v <- R[[k]][[b]]
      b=(b+1)
    }
    R[[k]][[b]] <- v + r*(TFun[[k]](v,0) - v)
  }
  pred <- R[[p]][[length(R[[p]])]]
  return(pred)
}

CAT <- function(Q, Method){
  
  if(Method=="FromMean"){
    ref <- rowMeans(Q)
    Tf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(ref, Q[,i], method = "natural")})
    ITf <- lapply(c(1:dim(Q)[2]), function(i){splinefun(Q[,i], ref, method = "natural")})
    grid <- ref
  }else if(Method=="FromDifference"){
    Tf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i], Q[,i+1], method = "natural")})
    ITf <- lapply(c(1:(dim(Q)[2]-1)), function(i){splinefun(Q[,i+1], Q[,i], method = "natural")})
    grid <- Q[, dim(Q)[2]]
  }else{
    stop("invalid method!")
  }

  Tv <- sapply(c(1:length(Tf)), function(i){return(Tf[[i]](grid, 0))})
  ITv <- sapply(c(1:length(ITf)), function(i){return(ITf[[i]](grid, 0))})
  
  n <- dim(Tv)[2]
  d <-  dim(Tv)[1]
  
  # case when alpha > 0
  M1 <- Tv - matrix(rep(grid, n), d, n)
  cov1 <- rowSums(M1[,1:(n-1)]*M1[,2:n])
  cov0 <- rowSums(M1[,1:(n-1)]*M1[,1:(n-1)])
  alpha1 <- cov1/cov0
  if(sum(is.na(alpha1))>0){
    alpha1 <- na.approx(alpha1)
  }
  alpha1 <- sapply(alpha1, function(a){
    if(a<0){return(0)}
    else{return(a)}
  })
  l1 = rowSums((M1[,2:n]-alpha1*M1[,1:(n-1)])^2)
  
  # case when alpha <0
  M2 <- matrix(rep(grid, n), d, n) - ITv
  v1 <- rowSums(M2[,1:(n-1)]*M1[,2:n])
  v0 <- rowSums(M2[,1:(n-1)]*M2[,1:(n-1)])
  alpha2 <- v1/v0
  if(sum(is.na(alpha2))>0){
    alpha2 <- na.approx(alpha2)
  }
  alpha2 <- sapply(alpha2, function(a){
    if(a>0){return(0)}
    else{return(a)}
  })
  l2 = rowSums((M1[,2:n]-alpha2*M2[,1:(n-1)])^2)
  
  result <- sapply(c(1:d), function(i){
    if(l1[i] <= l2[i]){
      return(c(alpha1[i], grid[i] + alpha1[i]*(Tv[i,n] - grid[i]))) 
    }else{
      pred = 
        return(c(alpha2[i], grid[i] + alpha2[i]*(grid[i] - ITv[i,n]))) 
    }
  })
  
  
  alpha<- result[1,]
  pred <- result[2,]
  
  if (is.unsorted(pred)){
    pred <- map2qt(pred)
  }
  
  return(list("alpha" = alpha, "pred"=pred))
}

map2boundary <- function(Logfit, sup) {
  if (!is.unsorted(Logfit + sup))
    return (1)
  eta0 <- 1
  eta1 <- 0.5
  while (abs(eta0 - eta1) > 1e-6 | is.unsorted(eta1 * Logfit + sup)) {
    if (is.unsorted(eta1 * Logfit + sup)) {
      tmp <- eta1
      eta1 <- eta1 - (eta0 - eta1) / 2
      eta0 <- tmp
    } else {
      eta1 <- (eta0 + eta1) / 2
    }
    #print(c(eta0,eta1))
  }
  return (eta1)
}

map2qt <- function ( y, lower = NULL, upper = NULL ) {
  m <- length(y)
  A <- cbind(diag(m), rep(0,m)) + cbind(rep(0,m), -diag(m))
  if (!is.null(upper) & !is.null(lower)) {
    b0 <- c(lower, rep(0,m-1), -upper)
  } else if(!is.null(upper)) {
    A <- A[,-1]
    b0 <- c(rep(0,m-1), -upper)
  } else if(!is.null(lower)) {
    A <- A[,-ncol(A)]
    b0 <- c(lower,rep(0,m-1))
  } else {
    A <- A[,-c(1,ncol(A))]
    b0 <- rep(0,m-1)
  }
  Pmat <- as(diag(m), "sparseMatrix")
  Amat <- as(t(A), "sparseMatrix")
  res <- do.call(
    osqp::solve_osqp,
    list(P=Pmat, q= -y, A=Amat, l=b0, pars = osqp::osqpSettings(verbose = FALSE))
  )
  sort(res$x)
}
