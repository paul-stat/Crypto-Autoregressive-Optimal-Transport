######Crypto Autoregressive Optimal Transport Code (main)
###The file "ATM_functions.R" provides R implementations for the ATM and CAT models

###main definitions
length_dSup=51 #support of the density when creating bins
length_qSup=31
qSup = seq(0, 1, length.out = length_qSup) #supporto of quantiles to define quantile functions

#libraries
library(frechet)
library(tidyr)
library(zoo)
library(MASS)
library(readxl)
library(RFPCA)
library(pracma)
library(fdapace)
library(ftsa)
library(fdadensity)
library(dplyr)
library(readr)
library(purrr)

#The file "ATM_functions.R" provides R implementations for the ATM models
source("ATM_functions.R")

# Create file names
years <- 2014:2024
file_names <- paste0("Bitstamp_BTCUSD_", years, "_minute.csv") #for Ethereum data, substitute with "Bitstamp_ETHUSD_"

# Read all files
all_data <- map_df(file_names, ~{
  tryCatch({
    read_csv(.x, skip = 1, col_types = cols( #skip=1 skips 1st row
      date = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
      close = col_double()
    ))
  }, error = function(e) {
    warning(paste("Error reading file:", .x))
    return(NULL)
  })
})

# Create myindexes dataframe
merged_data <- all_data %>%
  arrange(date) %>%
  distinct(date, .keep_all = TRUE)

#select only closing prices and volumes
myindex=merged_data[,c(7,8)]
Tbig=dim(myindex)[1]
myreturns=log(myindex[2:Tbig,1])-log(myindex[1:(Tbig-1),1])
myindex=myreturns #select returns to perform the analysis

# Function to separate the data into day, hour, and minute
mydate=merged_data$date

# Extract month, day, and minute
library(lubridate)
#split
partedata <- as.Date(mydate)
month <- month(mydate)
day_31 <- day(mydate)

#forming days
date <- ymd(partedata)
day <- yday(date) # so Jan 1 = day 1
cumulative_days=0 

#manage changing year
yr <- seq(2014, 2024, by=1)
year <- year(date)

# Loop through the years and modify the day count
for (i in 2:length(day)) {
  # If the year changes, increase the cumulative days by 365 or 366
  if (year[i] != year[i - 1]) {
    # Account for leap years (366 days) vs regular years (365 days)
    cumulative_days <- cumulative_days + ifelse(leap_year(year[i - 1]), 366, 365)
  }
  
  # Add cumulative days to the day count
  day[i] <- day[i] + cumulative_days
}

# Extract the number of unique dates
no_days <- length(unique(date))
mnth <- seq(1, 12, by=1)
dy <- seq(1, no_days, by=1) #dy <- seq(1, no_days, by=1)

dens <- vector(mode = "list", length = length(dy)) #container of density values
l <- min(myindex, na.rm = TRUE)
u <- max(myindex, na.rm = TRUE)

d <- vector("list", length = length(dy)) #d=n.rows di Q (matrice le cui colonne sono quantile functions sulla stessa grid (d,T))
bw <- (u-l)/10 #bandwith
dSup <- seq(l, u, length.out=length_dSup) #supporto della densità

#set aggregatore, dy = daily
aggregatore=dy

##for loop per aggregatore==dy
for(i in c(1:length(aggregatore))){
  tryCatch({
    index <- which(day %in% c(aggregatore[i])) 
    temp <- myindex[index,]
    x <- temp
    index <- which(!is.na(x))
    x <- x[index]
    
    d[[i]] <- frechet::CreateDensity(y=x, optns = list(userBwMu = bw, outputGrid=dSup))
  }, error = function(e) {
    if (i > 1) {
      # If an error occurs and it's not the first iteration, assign the previous value
      d[[i]] <- d[[i-1]]
      cat("Error caught for i =", i, ". Assigned previous value.\n")
    } else {
      # If it's the first iteration and an error occurs, we can't assign a previous value
      cat("Error caught for i =", i, ". No previous value to assign.\n")
    }
  })
  
  print(i)
}

#converting Densities to Quantile Functions
q <- vector("list", length = length(aggregatore))

last_valid_q <- NULL

for (i in 1:length(aggregatore)) {
  tryCatch({
    q[[i]] <- fdadensity::dens2quantile(dens = d[[i]]$y, dSup = dSup, qSup = qSup)
    if (!is.null(q[[i]]) && !any(is.na(q[[i]]))) {
      last_valid_q <- q[[i]]
    }
  }, error = function(e) {
    cat("Error caught for i =", i, ".\n")
  })
  
  if (is.null(q[[i]]) || any(is.na(q[[i]]))) {
    if (!is.null(last_valid_q)) {
      q[[i]] <- last_valid_q
      cat("Assigned last valid value for i =", i, ".\n")
    } else {
      cat("No valid value to assign for i =", i, ".\n")
    }
  }
  
  print(i)
}

# After the loop, fill any remaining NAs with the last non-NA value
last_valid_q <- NULL
for (i in 1:length(q)) {
  if (!is.null(q[[i]]) && !any(is.na(q[[i]]))) {
    last_valid_q <- q[[i]]
  } else if (!is.null(last_valid_q)) {
    q[[i]] <- last_valid_q
    cat("Filled NA with last valid value for i =", i, ".\n")
  }
}

# If there are still NAs at the beginning, fill forward
#first_valid_index <- which(!sapply(q, is.null) & !sapply(q, function(x) any(is.na(x))))[1]
#if (!is.na(first_valid_index) && first_valid_index > 1) {
#  for (i in (first_valid_index-1):1) {
#    q[[i]] <- q[[first_valid_index]]
#    cat("Filled forward NA at the beginning for i =", i, ".\n")
#  }
#}


#I assign to Q the values of Y (quantile function)
index <- c(1:(length(aggregatore)))
Q <- do.call(cbind, q[index])

#Recursive rolling experiment 
rw=365 #rolling window, same granularity as aggregatore
T_aggregatore=length(aggregatore) #n. units aggregatore totali
target_previsioni=matrix(NA,(T_aggregatore-rw),length(qSup))#prediction container

#contenitori (containers of results)
#atm
wd_err_atm=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_atm=rep(NA,(T_aggregatore-rw)) #container alpha
previsioni_atm=matrix(NA,(T_aggregatore-rw),length(qSup))#container predictions
Tloss_atm=rep(NA,(T_aggregatore-rw))
Lplus_atm=rep(NA,(T_aggregatore-rw))
Lminus_atm=rep(NA,(T_aggregatore-rw))

#atm_d
wd_err_atm_d=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_atm_d=rep(NA,(T_aggregatore-rw))
previsioni_atm_d=matrix(NA,(T_aggregatore-rw),length(qSup))#container predictions
Tloss_atm_d=rep(NA,(T_aggregatore-rw))
Lplus_atm_d=rep(NA,(T_aggregatore-rw))
Lminus_atm_d=rep(NA,(T_aggregatore-rw))

#cat
wd_err_cat=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_cat=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))
previsioni_cat=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))#container predictions

#cat_d
wd_err_cat_d=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_cat_d=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))
previsioni_cat_d=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))#container predictions


#for loop
for(i in 1:(T_aggregatore-rw)){
  tryCatch({
    atm <- ATM(Q[,1:(i+rw)], p=1, eta =0.01, Method = "FromMean")
    atm_d <- ATM(Q[,1:(i+rw)], p=1, eta =0.01, Method = "FromDifference")
    cat <- CAT(Q[c(2:dim(Q)[1]),1:(i+rw)], Method = "FromMean")
    cat_d <- CAT(Q[c(2:dim(Q)[1]),1:(i+rw)], Method = "FromDifference")
    #d_pred <- frechet:::qf2pdf(atm$pred, qSup, optns = list(userBwMu = bw, outputGrid=dSup))
    #lines(d_pred$x, d_pred$y, col="red", lty=1, lwd=2)
    target <- q[[i+1]]
    target_previsioni[i,]=target
    Lplus_atm[i]=atm$L_plus
    Lminus_atm[i]=atm$L_minus
    Lplus_atm_d[i]=atm_d$L_plus
    Lminus_atm_d[i]=atm_d$L_minus
    Tloss_atm[i]=atm$TrainingLoss
    Tloss_atm_d[i]=atm_d$TrainingLoss
    previsioni_atm[i,]=atm$pred
    previsioni_atm_d[i,]=atm_d$pred
    previsioni_cat[i,]=cat$pred
    previsioni_cat_d[i,]=cat_d$pred
    wd_err_atm[i]=sqrt(trapzRcpp(qSup, (atm$pred-target)^2)) 
    wd_err_atm_d[i]=sqrt(trapzRcpp(qSup, (atm_d$pred-target)^2)) 
    wd_err_cat[i]=sqrt(trapzRcpp(qSup, (c(min(cat$pred),cat$pred)-target)^2))
    wd_err_cat_d[i]=sqrt(trapzRcpp(qSup, (c(min(cat_d$pred),cat$pred)-target)^2))
    alpha_atm[i]=atm$alpha
    alpha_atm_d[i]=atm_d$alpha
    alpha_cat[i,]=cat$alpha
    alpha_cat_d[i,]=cat_d$alpha
  }, error = function(e) {
    cat("Error caught for i =", i, ". Using previous iteration's values.\n")
    if (i > 1) {
      Lplus_atm[i]=Lplus_atm[i-1]
      Lminus_atm[i]=Lminus_atm[i-1]
      Lplus_atm_d[i]=Lplus_atm_d[i-1]
      Lminus_atm_d[i]=Lminus_atm_d[i-1]
      Tloss_atm[i]=Tloss_atm[i-1]
      Tloss_atm_d[i]=Tloss_atm_d[i-1]
      previsioni_atm[i,]=previsioni_atm[i-1]
      previsioni_atm_d[i,]=previsioni_atm_d[i-1]
      previsioni_cat[i,]=previsioni_cat[i-1]
      previsioni_cat_d[i,]=previsioni_cat_d[i-1]
      wd_err_atm[i] <- wd_err_atm[i-1]
      wd_err_atm_d[i] <- wd_err_atm_d[i-1]
      wd_err_cat[i] <- wd_err_cat[i-1]
      wd_err_cat_d[i] <- wd_err_cat_d[i-1]
      alpha_atm[i] <- alpha_atm[i-1]
      alpha_atm_d[i] <- alpha_atm_d[i-1]
      alpha_cat[i,] <- alpha_cat[i-1,]
      alpha_cat_d[i,] <- alpha_cat_d[i-1,]
    } else {
      cat("Error occurred in the first iteration. Unable to use previous values.\n")
    }
  })
  print(i) 
}



date_vector <- seq(from = as.Date("2014-01-01"), 
                   to = as.Date("2024-09-28"), 
                   by = "day")

#dataframes to store values

#Tloss_atm=Tloss_atm[2:length(Tloss_atm)]
#Tloss_atm_d=Tloss_atm_d[2:length(Tloss_atm_d)]
lplusminusTloss=data.frame(Lplus_atm,Lplus_atm_d,Lminus_atm,Lminus_atm_d,Tloss_atm,Tloss_atm_d)
forecasts=data.frame(previsioni_atm,previsioni_atm_d,previsioni_cat,previsioni_cat_d,target_previsioni)
errors=data.frame(date_vector[rw:(T_aggregatore-1)],wd_err_atm,wd_err_atm_d,wd_err_cat,wd_err_cat_d)
alphas_atm=data.frame(date_vector[rw:(T_aggregatore-1)],alpha_atm,alpha_atm_d)
alphas_cat=data.frame(date_vector[rw:(T_aggregatore-1)],alpha_cat,alpha_cat_d)
target_full=rbind(t(Q[,1:rw]),target_previsioni)

#save
write.csv(lplusminusTloss,"lplusminusTloss_exprw.csv")
write.csv(forecasts,"forecasts_exprw.csv")
write.csv(errors,"errors_exprw.csv")
write.csv(alphas_atm,"alphas_atm_exprw.csv")
write.csv(alphas_cat,"alphas_cat_exprw.csv")
write.csv(target_full,"ret_targetfull.csv")

#save(list = ls(all.names = TRUE), file = "environment_ret_exprw.RData")

#####  VOLATILITY DISTRIBUTION ####

#compute volatility from HLOC data
library(TTR)
ohlc=all_data[,4:7]
vGK <- volatility(ohlc, calc="garman") #can be substituted with other volatility estimations, e.g. "rogers.satchell"
tagliovola=12


#check NA
sum(is.na(vGK))
#fill potential missing NA
vGK <- na.locf(vGK, na.rm = FALSE)
vGK=vGK[tagliovola:length(vGK)]#cut vola

dens <- vector(mode = "list", length = length(dy)) #container of density values
l <- min(vGK, na.rm = TRUE)
u <- max(vGK, na.rm = TRUE)

d <- vector("list", length = length(dy)) 
bw <- (u-l)/10 #bandwith
dSup <- seq(l, u, length.out=length_dSup) 

#set aggregatore, dy = daily
aggregatore=dy

##for loop per aggregatore==dy
for(i in c(1:length(aggregatore))){
  tryCatch({
    index <- which(day %in% c(aggregatore[i])) 
    temp <- myindex[index,]
    x <- temp
    index <- which(!is.na(x))
    x <- x[index]
    
    d[[i]] <- frechet::CreateDensity(y=x, optns = list(userBwMu = bw, outputGrid=dSup))
  }, error = function(e) {
    if (i > 1) {
      # If an error occurs and it's not the first iteration, assign the previous value
      d[[i]] <- d[[i-1]]
      cat("Error caught for i =", i, ". Assigned previous value.\n")
    } else {
      # If it's the first iteration and an error occurs, we can't assign a previous value
      cat("Error caught for i =", i, ". No previous value to assign.\n")
    }
  })
  
  print(i)
}

#converting Densities to Quantile Functions
q <- vector("list", length = length(aggregatore))

last_valid_q <- NULL

for (i in 1:length(aggregatore)) {
  tryCatch({
    q[[i]] <- fdadensity::dens2quantile(dens = d[[i]]$y, dSup = dSup, qSup = qSup)
    if (!is.null(q[[i]]) && !any(is.na(q[[i]]))) {
      last_valid_q <- q[[i]]
    }
  }, error = function(e) {
    cat("Error caught for i =", i, ".\n")
  })
  
  if (is.null(q[[i]]) || any(is.na(q[[i]]))) {
    if (!is.null(last_valid_q)) {
      q[[i]] <- last_valid_q
      cat("Assigned last valid value for i =", i, ".\n")
    } else {
      cat("No valid value to assign for i =", i, ".\n")
    }
  }
  
  print(i)
}

# After the loop, fill any remaining NAs with the last non-NA value
last_valid_q <- NULL
for (i in 1:length(q)) {
  if (!is.null(q[[i]]) && !any(is.na(q[[i]]))) {
    last_valid_q <- q[[i]]
  } else if (!is.null(last_valid_q)) {
    q[[i]] <- last_valid_q
    cat("Filled NA with last valid value for i =", i, ".\n")
  }
}

# If there are still NAs at the beginning, fill forward
#first_valid_index <- which(!sapply(q, is.null) & !sapply(q, function(x) any(is.na(x))))[1]
#if (!is.na(first_valid_index) && first_valid_index > 1) {
#  for (i in (first_valid_index-1):1) {
#    q[[i]] <- q[[first_valid_index]]
#    cat("Filled forward NA at the beginning for i =", i, ".\n")
#  }
#}

#I assign to Q the values of Y
index <- c(1:(length(aggregatore)))
Q <- do.call(cbind, q[index])

#Recursive rolling experiment
rw=365 #rolling window, same granularity as aggregatore
T_aggregatore=length(aggregatore) #n. unità aggregatore totali
target_previsioni=matrix(NA,(T_aggregatore-rw),length(qSup))#container predictions

#contenitori (containers of results)
#atm
wd_err_atm=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_atm=rep(NA,(T_aggregatore-rw)) #container alpha
previsioni_atm=matrix(NA,(T_aggregatore-rw),length(qSup))#container predictions
Tloss_atm=rep(NA,(T_aggregatore-rw))
Lplus_atm=rep(NA,(T_aggregatore-rw))
Lminus_atm=rep(NA,(T_aggregatore-rw))

#atm_d
wd_err_atm_d=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_atm_d=rep(NA,(T_aggregatore-rw))
previsioni_atm_d=matrix(NA,(T_aggregatore-rw),length(qSup))#container predictions
Tloss_atm_d=rep(NA,(T_aggregatore-rw))
Lplus_atm_d=rep(NA,(T_aggregatore-rw))
Lminus_atm_d=rep(NA,(T_aggregatore-rw))

#cat
wd_err_cat=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_cat=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))
previsioni_cat=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))#container predictions

#cat_d
wd_err_cat_d=rep(NA,(T_aggregatore-rw)) #Wasserstein distance between observed and predicted distribution
alpha_cat_d=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))
previsioni_cat_d=matrix(NA,(T_aggregatore-rw),(length(qSup)-1))#container predictions

#for loop
for(i in 1:(T_aggregatore-rw)){
  tryCatch({
    atm <- ATM(Q[,1:(i+rw)], p=1, eta =0.01, Method = "FromMean")
    atm_d <- ATM(Q[,1:(i+rw)], p=1, eta =0.01, Method = "FromDifference")
    cat <- CAT(Q[c(2:dim(Q)[1]),1:(i+rw)], Method = "FromMean")
    cat_d <- CAT(Q[c(2:dim(Q)[1]),1:(i+rw)], Method = "FromDifference")
    target <- q[[i+1]]
    target_previsioni[i,]=target
    Lplus_atm[i]=atm$L_plus
    Lminus_atm[i]=atm$L_minus
    Lplus_atm_d[i]=atm_d$L_plus
    Lminus_atm_d[i]=atm_d$L_minus
    Tloss_atm[i]=atm$TrainingLoss
    Tloss_atm_d[i]=atm_d$TrainingLoss
    previsioni_atm[i,]=atm$pred
    previsioni_atm_d[i,]=atm_d$pred
    previsioni_cat[i,]=cat$pred
    previsioni_cat_d[i,]=cat_d$pred
    wd_err_atm[i]=sqrt(trapzRcpp(qSup, (atm$pred-target)^2)) 
    wd_err_atm_d[i]=sqrt(trapzRcpp(qSup, (atm_d$pred-target)^2)) 
    wd_err_cat[i]=sqrt(trapzRcpp(qSup, (c(min(cat$pred),cat$pred)-target)^2))
    wd_err_cat_d[i]=sqrt(trapzRcpp(qSup, (c(min(cat_d$pred),cat$pred)-target)^2))
    alpha_atm[i]=atm$alpha
    alpha_atm_d[i]=atm_d$alpha
    alpha_cat[i,]=cat$alpha
    alpha_cat_d[i,]=cat_d$alpha
  }, error = function(e) {
    cat("Error caught for i =", i, ". Using previous iteration's values.\n")
    if (i > 1) {
      Lplus_atm[i]=Lplus_atm[i-1]
      Lminus_atm[i]=Lminus_atm[i-1]
      Lplus_atm_d[i]=Lplus_atm_d[i-1]
      Lminus_atm_d[i]=Lminus_atm_d[i-1]
      Tloss_atm[i]=Tloss_atm[i-1]
      Tloss_atm_d[i]=Tloss_atm_d[i-1]
      previsioni_atm[i,]=previsioni_atm[i-1]
      previsioni_atm_d[i,]=previsioni_atm_d[i-1]
      previsioni_cat[i,]=previsioni_cat[i-1]
      previsioni_cat_d[i,]=previsioni_cat_d[i-1]
      wd_err_atm[i] <- wd_err_atm[i-1]
      wd_err_atm_d[i] <- wd_err_atm_d[i-1]
      wd_err_cat[i] <- wd_err_cat[i-1]
      wd_err_cat_d[i] <- wd_err_cat_d[i-1]
      alpha_atm[i] <- alpha_atm[i-1]
      alpha_atm_d[i] <- alpha_atm_d[i-1]
      alpha_cat[i,] <- alpha_cat[i-1,]
      alpha_cat_d[i,] <- alpha_cat_d[i-1,]
    } else {
      cat("Error occurred in the first iteration. Unable to use previous values.\n")
    }
  })
  print(i) 
}



#dataframes to store values
lplusminusTloss=data.frame(Lplus_atm,Lplus_atm_d,Lminus_atm,Lminus_atm_d,Tloss_atm,Tloss_atm_d)
forecasts=data.frame(previsioni_atm,previsioni_atm_d,previsioni_cat,previsioni_cat_d,target_previsioni)
errors=data.frame(date[rw:(T_aggregatore-1)],wd_err_atm,wd_err_atm_d,wd_err_cat,wd_err_cat_d)
alphas_atm=data.frame(date[rw:(T_aggregatore-1)],alpha_atm,alpha_atm_d)
alphas_cat=data.frame(date[rw:(T_aggregatore-1)],alpha_cat,alpha_cat_d)

target_full=rbind(t(Q[,1:rw]),target_previsioni)

#save
write.csv(lplusminusTloss,"vola_lplusminusTloss_exprw.csv")
write.csv(forecasts,"vola_forecasts_exprw.csv")
write.csv(errors,"vola_errors_exprw.csv")
write.csv(alphas_atm,"vola_alphas_atm_exprw.csv")
write.csv(alphas_cat,"vola_alphas_cat_exprw.csv")
write.csv(target_full,"vola_targetfull.csv")


#save(list = ls(all.names = TRUE), file = "environment_vola_exprw.RData")
