#***************************************************************************************************************
#***************************************************************************************************************
#
# Authors: Santiago Gil-Begue, Pedro Larrañaga and Concha Bielza
#
# Thanks to: Sergio Luengo-Sanchez
#
# Notes: This code is part of the publication with name 'Multi-dimensional Bayesian network classifier trees'
#        published in Proceedings of the 19th International Conference on Intelligent Data Engineering and
#        Automated Learning, Lecture Notes in Artificial Intelligence, Springer (2018).
#
# Further information: sgil@fi.upm.es
#
#***************************************************************************************************************
#***************************************************************************************************************

{
#install.packages("BiocManager")
#BiocManager::install(c("igraph", "graph", "RBGL", "Rgraphviz"))
#install.packages("bnlearn", dependencies=TRUE)
library("bnlearn")
#install.packages("utiml")
library("utiml")
#install.packages("e1071")
#install.packages("randomForest")
#install.packages("FSelector",dependencies=TRUE)
library("FSelector")
#install.packages("foreign")
library("foreign")
#install.packages("arules")
library("arules")
#install.packages("mldr.datasets")
library("mldr.datasets")
#install.packages("foreach")
library("foreach")
#install.packages("caret")
library("caret")
#install.packages("doParallel")
library("doParallel")
registerDoParallel(makeCluster(detectCores()))
}

{ # Define all functions together
  
################################################################################################################
##########################                                                            ##########################
##########################              PERFORMANCE EVALUATION MEASURES               ##########################
##########################                                                            ##########################
################################################################################################################

##
# <test_set> and <out> : data.frame
# <classes> : character (vector)
# 
# Calculates the multi-label performance over the binary class variables <classes>,
# such that <test_set> are the true values of the variables and <out> its predictions
#
# Returns a list with:
#  - Key <mlconfmat> : mlconfmat
#      Confusion matrices for each label
#  - Key <measures> : numeric (vector)
#      Several multi-label performance evaluation measures
#
# See <utiml> package
##
test_multilabel <- function(test_set, out, classes) {
  # Convert from data.frames to <utiml> required objects (mldr and mlresult)
  if (!is.na(as.integer(test_set[1, classes[1]]))) { # 0 1
    true <- as.data.frame(apply(test_set[, classes], 2, function(x) as.integer(as.character(x))))
  }
  else { # TRUE FALSE
    true <- as.data.frame(apply(test_set[, classes], 2, function(x) as.integer(as.logical(x))))
  }
  true_mldr <- mldr_from_dataframe(true, labelIndices = 1:length(classes))
  if (!is.na(as.integer(out[1, 1]))) {
    out <- as.data.frame(apply(out, 2, function(x) as.integer(as.character(x))))
  }
  else {
    out <- as.data.frame(apply(out, 2, function(x) as.integer(as.logical(x))))
  }
  out_mlresult <- as.mlresult(out, probability=TRUE)
  # Confusion matrices
  mlconfmat <- multilabel_confusion_matrix(true_mldr, out_mlresult)
  # Multi-label performance evaluation measures. All possible are given
  measures <- multilabel_evaluate(mlconfmat, measures=c("all"))
  return(list("mlconfmat"=mlconfmat, "measures"=measures))
}

###
# <test_set> and <out> : data.frame
# <classes> : character (vector)
# 
# Calculates the multi-dimensional performance over the class variables <classes>,
# such that <test_set> are the true values of the variables and <out> its predictions
#
# Returns the multi-dimensional performance in a list structure such that:
#  - Key <global> : numeric
#      Global accuracy over the d-dimensional class
#  - Key <average> : numeric
#      Average accuracy over the d classes
#  - Key <per_class> : numeric (vector)
#      Marginal accuracy on each class variable, given in the same order than in <classes>
###
test_multidimensional <- function(test_set, out, classes) {
  # Step needed before '==' operator
  true <- as.data.frame(sapply(test_set[, classes], as.character), stringsAsFactors=FALSE)
  out <- as.data.frame(sapply(out, as.character), stringsAsFactors=FALSE)
  match <- true == out
  exact_match <- rep(TRUE, nrow(test_set))
  per_class <- vector("numeric", length=length(classes))
  for (i in 1:length(classes)) {
    per_class[i] <- mean(match[,i])
    exact_match <- exact_match & match[,i]
  }
  # Average accuracy
  average_accuracy <- mean(per_class)
  # Global accuracy
  global_accuracy <- mean(exact_match)
  return(list("global"=global_accuracy, "average"=average_accuracy, "per_class"=per_class))
}

###
# <test_set> and <out> : data.frame
# <classes> : character (vector)
# 
# Calculates the confusion matrix for each class variable in <classes>,
# such that <test_set> are the true values of the variables and <out> its predictions
#
# Returns all the confusion matrices in a list structure such that the keys are the values of <classes>
#  - Key <class> in <classes> : matrix
#      Confusion matrix of the class variable <class>
###
multidimensional_confusion_matrix <- function(test_set, out, classes) {
  confusion_matrices <- list()
  for (i in 1:length(classes)) {
    class <- classes[i]
    values <- unique(c(levels(test_set[,class]), levels(out[,class])))
    confusion_matrix <- matrix(, length(values), length(values))
    dimnames(confusion_matrix) <- list(sapply(values, function(x) paste0("Predicted ", x)),
                                       sapply(values, function(x) paste0("True ", x)))
    for (j in 1:length(values)) {
      value <- values[j]
      confusion_matrix[,paste0("True ",value)] <-
        sapply(values, function(x) sum(test_set[,class] == value & out[,class] == x))
    }
    confusion_matrices[[class]] <- confusion_matrix
  }
  return(confusion_matrices)
}

################################################################################################################
##########################                                                            ##########################
##########################                    PREDICT WITH MODELS                     ##########################
##########################                                                            ##########################
################################################################################################################

###
# <MBC> : c('bn.fit', 'bn.fit.dnet')
# <case> : data.frame (with just one row)
# <classes> and <features> : character (vector)
#
# Predicts the class variables <classes> given the values of the features variables <features>
# of the instance <case> by using the multi-dimensional Bayesian network classifier <MBC>
#
# Allows missing data
#
# Returns the most probable explanation (MPE) of the class variables
#   <out> : character (vector)
###
predict_MBC_case <- function(MBC, case, classes, features) {
  net_ev <- gRain::setEvidence(as.grain(MBC), evidence=lapply(case[features], function(x) as.character(x)))
  res <- gRain::querygrain(net_ev, nodes=classes, type="joint")
  # MPE (0-1 loss function)
  inds <- arrayInd(which.max(res), dim(res))
  out <- mapply(function(dimnames, ind) dimnames[ind], dimnames(res), inds)
  return(out)
}

###
# The same as <predict_MBC_case> but...
#
# <test_set> : data.frame (multiple rows=cases)
#
# Predicts the class variables of all the cases in <test_set> data set
#
# Allows missing data
#
# <out> : data.frame (nrow(out) == nrow(test_set))
###
predict_MBC_dataset <- function(MBC, test_set, classes, features) {
  out <- test_set[,classes] # To maintain factors format
  for (i in 1:nrow(test_set)) {
    # <foo> classes may be given in different order than those in <out>
    foo <- predict_MBC_case(MBC, test_set[i,], classes, features)
    for (j in 1:length(classes)) {
      clase <- classes[j]
      out[i,clase] <- foo[clase]
    }
  }
  return(out)
}

## BRUTE FORCE APPROACHES, JUST FOR PROBLEMS WITH FEW CLASS VARIABLES

##
# The same as predict_MBC_dataset, but...
#   - Less computational time
#   + More memory management, in relation to the class space dimension
#
# Does not allow missing data
##
predict_MBC_dataset_fast <- function(MBC, test_set, classes, features) {
  options(warn = -1) # Supress warnings
  out <- test_set[,classes]
  # Joint class configurations
  classes_levels <- lapply(classes, function(x) attributes(MBC[[x]]$prob)$dimnames[[1]])
  names(classes_levels) <- classes
  classes_joint <- expand.grid(classes_levels)
  # An instance + joint class configurations
  matrix_MPE <- cbind(matrix(ncol=length(features), nrow=nrow(classes_joint), dimnames=list(NULL, features)),
                      classes_joint)
  # Obtain MPE as argmax p(classes, features), what is the same as argmax p(classes | features)
  for (i in 1:nrow(test_set)) {
    matrix_MPE[,features] <- test_set[i, features]
    index_MPE <- which.max(logLik(MBC, matrix_MPE, by.sample=TRUE))
    out[i,] <- matrix_MPE[index_MPE, classes]
  }
  return(out)
}

##
# The same as predict_MBC_dataset_fast, but...
#   - Less computational time
#   - Much more memory management
#
# Does not allow missing data
##
predict_MBC_dataset_veryfast <- function(MBC, test_set, classes, features) {
  # Divided in two functions because <obtain_big_matrix_MPE> can be computed just once in other algorithms
  big_matrix_MPE <- obtain_big_matrix_MPE(MBC, test_set, classes, features)
  return(predict_MBC_big_matrix_MPE(MBC, big_matrix_MPE$matrix, big_matrix_MPE$joint))
}

obtain_big_matrix_MPE <- function(MBC, test_set, classes, features) {
  # Joint class configurations
  classes_levels <- lapply(classes, function(x) attributes(MBC[[x]]$prob)$dimnames[[1]])
  names(classes_levels) <- classes
  classes_joint <- expand.grid(classes_levels)
  I <- nrow(classes_joint)
  # All instances + joint class configurations
  big_matrix_MPE <- cbind(test_set[rep(1:nrow(test_set), each = I), features],
                          classes_joint[rep(seq_len(I), nrow(test_set)),])
  # Fix bug when there is one feature. The column won't be named as the feature, do it manually
  if (length(features) == 1) { colnames(big_matrix_MPE)[1] <- features }
  return(list("matrix"=big_matrix_MPE, "joint"=classes_joint))
}

predict_MBC_big_matrix_MPE <- function(MBC, big_matrix_MPE, classes_joint) {
  options(warn = -1) # Supress warnings
  I <- nrow(classes_joint)
  # Obtain MPE as argmax p(classes, features), what is the same as argmax p(classes | features)
  likelihood <- logLik(MBC, big_matrix_MPE, by.sample=TRUE)
  indexes_MPE <- sapply(0:(nrow(big_matrix_MPE)/I-1), function(x) which.max(likelihood[(x*I+1):(x*I+I)]))
  return(classes_joint[indexes_MPE,])
}

###
# The same as <predict_MBC_case> but using an MBCTree <MBCTree> as the classifier
###
predict_MBCTree_case <- function(MBCTree, case, classes, features) {
  # Reach the corresponding MBC leaf
  MBCTree_aux <- MBCTree
  features_used <- list()
  while (MBCTree_aux$leaf == FALSE) {
    features_used <- c(features_used, MBCTree_aux$feature)
    lab <- case[,MBCTree_aux$feature]
    MBCTree_aux <- MBCTree_aux$MBC_split[[lab]]
  }
  # Predict the case with the reached MBC leaf
  features_rest <- features[!features %in% features_used]
  return(predict_MBC_case(MBCTree_aux$MBC, case, classes, features_rest))
}

###
# The same as <predict_MBC_dataset> but using an MBCTree <MBCTree> as the classifier
###
predict_MBCTree_dataset <- function(MBCTree, test_set, classes, features) {
  out <- test_set[,classes] # To maintain factors format
  for (i in 1:nrow(test_set)) {
    # <foo> classes may be given in different order than those in <out>
    foo <- predict_MBCTree_case(MBCTree, test_set[i,], classes, features)
    for (j in 1:length(classes)) {
      clase <- classes[j]
      out[i,clase] <- foo[clase]
    }
  }
  return(out)
}

###
# The same as <predict_MBC_dataset_veryfast> but using an MBCTree <MBCTree> as the classifier
#
# Returns a list with:
#   - Key <out> : data.frame
#       Predicted classes
#   - Key <true> : data.frame
#       True classes. Given because the order of the instances is modified from <test_set> because of efficiency
###
predict_MBCTree_dataset_veryfast <- function(MBCTree, test_set, classes, features) {
  if (MBCTree$leaf == TRUE) {
    true <- test_set[,classes]
    out <- predict_MBC_dataset_veryfast(MBCTree$MBC, test_set, classes, features)
  }
  else {
    out <- test_set[,classes] # To maintain factors format
    true <- out
    features_rest <- features[!features %in% MBCTree$feature]
    labels <- names(MBCTree$MBC_split)
    index <- 1
    for (i in 1:length(labels)) {
      lab <- labels[i]
      test_set_filtered <- test_set[test_set[,MBCTree$feature] == lab, c(classes,features_rest)]
      if (nrow(test_set_filtered) > 0) {
        index_prev <- index
        index <- index + nrow(test_set_filtered)
        result <- predict_MBCTree_dataset_veryfast(MBCTree$MBC_split[[lab]],
                   test_set_filtered, classes, features_rest)
        true[index_prev:(index-1),] <- result$true
        out[index_prev:(index-1),] <- result$out
      }
    }
  }
  return(list("true"=true, "out"=out))
}

################################################################################################################
##########################                                                            ##########################
##########################                        LEARN MODELS                        ##########################
##########################                                                            ##########################
################################################################################################################

###
# <training_set> : data.frame
# <classes> and <features> : character (vector)
#
# Learns an MBC from <training_set> data set with class variables <classes> and feature variables <features>
# in a filter way using the hill climbing algorithm and maximizing tbe BIC score.
# Bayesian method is used for the parameter estimation, Laplace rule is used for regularization.
#
# The algoithm starts with a full empty MBC, and it is updated in each itation with the addition, deletion
# or reversal that most improves the BIC score. It finishes when no arcs can be added, deleted or reversed
# such that the score improves.
#
# Returns the learned MBC
#  <MBC> : c('bn.fit', 'bn.fit.dnet')
###
learn_MBC <- function(training_set, classes, features) {
  # Black list of arcs from features to classes
  bl <- matrix(nrow=length(classes)*length(features), ncol=2, dimnames=list(NULL, c("from","to")))
  bl[,"from"] <- rep(features, each=length(classes))
  bl[,"to"] <- rep(classes, length(features))
  # Learn MBC structure
  net <- hc(training_set, blacklist=bl)
  # Fit CPTs
  MBC <- bn.fit(net, training_set, method="bayes", iss=1) # iss = 1 -> Laplace
  return(MBC)
}

learn_MBC_wrapper <- function(training_set, validation_set, classes, features, search_times=50, verbose=FALSE) {
  # Start from a empty graph
  MBC_best <- empty.graph(c(classes, features), num = 1)
  MBC_fit <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)
  # --- First part of <predict_MBC_dataset_veryfast>. Computed just once
  big_matrix_MPE <- obtain_big_matrix_MPE(MBC_fit, validation_set, classes, features)
  # Test it
  out <- predict_MBC_big_matrix_MPE(MBC_fit, big_matrix_MPE$matrix, big_matrix_MPE$joint)
  performance_best <- test_multidimensional(validation_set, out, classes)$global
  # Iteratively add/remove multiple arcs at each iteration that improve the global accuracy
  candidates <- MBC_possible_arcs(classes, features)
  for (i in 1:search_times) {
    if (verbose) { print(paste0("Epoch: ", i)) }
    MBC <- MBC_best
    arcs_changed <- sample(1:length(classes), 1)
    for (j in 1:arcs_changed) {
      arcs <- arcs(MBC)
      adding <- runif(1, 0.0, 1.0) > 0.2
      for (k in 1:length(features)) {
        arc <- candidates[sample(1:nrow(candidates), 1),]
        # Check if the arc is not in the Markov Blanket of any class variable
        if (adding) {
          if (arc["from"] %in% features) {
            interest <- FALSE
            for (j in 1:length(classes)) {
              if (arc["to"] %in% MBC$nodes[[classes[[j]]]]$children) {
                interest <- TRUE
                break
              }
            }
            if (!interest) { next }
          }
        }
        # Check if the arc is already present
        arc_present <- FALSE
        if (nrow(arcs) > 0) {
          for (j in 1:nrow(arcs)) {
            if (arcs[j,"from"] == arc["from"] & arcs[j,"to"] == arc["to"]) {
              arc_present <- TRUE
              break
            }
          }
        }
        if (adding & !arc_present) {
          MBC_aux <- set.arc(MBC, from=arc["from"], to=arc["to"], check.cycles=FALSE)
          if (acyclic(MBC_aux, directed=TRUE)) {
            MBC <- MBC_aux
            break
          }
        }
        else if (!adding & arc_present) {
          MBC <- drop.arc(MBC, from=arc["from"], to=arc["to"])
          break
        }
      }
    }
    # Evaluates the new MBC
    MBC_fit <- bn.fit(MBC, training_set, method="bayes", iss=1)
    out <- predict_MBC_big_matrix_MPE(MBC_fit, big_matrix_MPE$matrix, big_matrix_MPE$joint)
    performance <- test_multidimensional(validation_set, out, classes)$global
    if (performance > performance_best) {
      if (verbose) {
        print(paste0("Acc before: ", performance_best))
        print(paste0("Acc now: ", performance))
      }
      performance_best <- performance
      MBC_best <- MBC
    }
  }
  return(bn.fit(MBC_best, training_set, method = "bayes", iss = 1))
}

###
# <training_set> and <validation_set> : data.frame
# <classes> and <features> : character (vector)
#
# Learns an MBC from <training_set> and <validation_set> data sets with class variables <classes>
# and feature variables <features>. A greedy wrapper strategy is applied, such that it starts from
# an empty graph, and tries to iteratively add the arc that most improves the global accuracy.
# <training_set> is used for training the current MBC and <validation_set> to compute the accuracy
# improvements achieved with the additions of the arcs. It stops when no arc can be added
# such that an improvement is achieved.
# Bayesian method is used for the parameter estimation, Laplace rule is used for regularization.
#
# Returns the learned MBC
#  <MBC_fit_best> : c('bn.fit', 'bn.fit.dnet')
###
learn_MBC_wrapper2 <- function(training_set, validation_set, classes, features, verbose=FALSE) {
  # Start from an empty graph
  MBC_best <- empty.graph(c(classes, features), num = 1)
  MBC_fit <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)
  # --- First part of <predict_MBC_dataset_veryfast>. Computed just once
  big_matrix_MPE <- obtain_big_matrix_MPE(MBC_fit, validation_set, classes, features)
  # Test it
  out <- predict_MBC_big_matrix_MPE(MBC_fit, big_matrix_MPE$matrix, big_matrix_MPE$joint)
  performance_best <- test_multidimensional(validation_set, out, classes)$global
  performance_ant <- 0
  # Iteratively add the arc that most improves the global accuracy
  candidates <- MBC_possible_arcs(classes, features)
  while (performance_best > performance_ant) {
    scores <- rep(0, nrow(candidates))
    arcs <- arcs(MBC_best)
    for (i in 1:nrow(candidates)) {
      arc <- candidates[i,]
      # Check if the arc is not in the Markov Blanket of any class variable
      if (arc["from"] %in% features) {
        interest <- FALSE
        for (j in 1:length(classes)) {
          if (arc["to"] %in% MBC_best$nodes[[classes[[j]]]]$children) {
            interest <- TRUE
            break
          }
        }
        if (!interest) { next }
      }
    
      # Check if the arc is already present
      arc_present <- FALSE
      if (nrow(arcs) > 0) {
        for (j in 1:nrow(arcs)) {
          if (arcs[j,"from"] == arc["from"] & arcs[j,"to"] == arc["to"]) {
            arc_present <- TRUE
            break
          }
        }
      }
      if (arc_present) { next }
      
      # Check if the arc would involve a cycle
      MBC <- set.arc(MBC_best, from=arc["from"], to=arc["to"], check.cycles=FALSE)
      if (!acyclic(MBC, directed=TRUE)) { next }
      
      # Evaluates the arc addition
      MBC_fit <- bn.fit(MBC, training_set, method="bayes", iss=1)
      out <- predict_MBC_big_matrix_MPE(MBC_fit, big_matrix_MPE$matrix, big_matrix_MPE$joint)
      scores[i] <- test_multidimensional(validation_set, out, classes)$global
    }
    
    # Add the best arc, if any
    pos_best <- which.max(scores)
    performance_ant <- performance_best
    performance_best <- scores[pos_best]
    if (verbose) {
      print(candidates[pos_best,])
      print(paste0("Acc before: ", performance_ant))
      print(paste0("Acc now: ", performance_best))
    }
    
    if (performance_best > performance_ant) {
      arc_best <- candidates[pos_best,]
      MBC_best <- set.arc(MBC_best, from=arc_best["from"], to=arc_best["to"], check.cycles=FALSE)
    }
  }
  return(bn.fit(MBC_best, training_set, method = "bayes", iss = 1))
}

###
# <training_set> and <validation_set> : data.frame
# <classes> and <features> : character (vector)
#
# Learns an MBCTree from <training_set> and <validation_set> data sets with class
# variables <classes> and feature variables <features>. The algorithms follows a filter 
# approach guided by the BIC if <filter> is TRUE, and a wrapper approach guided by the
# global/average accuracy if <filter> is FALSE.
#
# In the case of a filter approach:
#  - <training_set> and <validation_set> are both used to learn and evaluate (BIC) MBCs
# In the case of a wrapper approach:
#  - <training_set> is used to learn MBCs
#  - <validation_set> is used to evaluate the MBCs so that the best split can be computed
#    + If measure="global", global accuracy is evaluated.
#    + If measure="average", average accuracy is evaluated.
#
# Returns the learned MBCTree as a recursive list such that:
#  + A leaf node is a list with:
#    - Key <MBC> : c('bn.fit', 'bn.fit.dnet')
#        MBC associated to the leaf node
#    - Key <leaf> : logical
#        A TRUE value meaning this is a leaf node
#    - Key <performance> : numeric
#        Global/average accuracy or BIC of the leaf MBC over the corresponding portion of data
#  + An internal node is a list with:
#    - Key <feature> : character
#        The feature variable that splits the data in the current internal node
#    - Key <MBC_split> : list
#        The possible values of the feature variable <feature> are the keys of the list
#        Each element is another list representing the sub-MBCTree associated to each child
#    - Key <leaf> : logical
#        A FALSE value meaning this is an internal node
#    - Key <performance> : numeric
#        Global/average accuracy or BIC of an MBC over the corresponding portion of data that would
#        have been placed insted of this internal node 
#    - Key <performance_split> : numeric
#        Global/average accuracy or BIC of the split MBCs on the best feature variable
#        (performance_split > performance) over the corresponding portion of data
###
learn_MBCTree <- function(training_set, validation_set, classes, features,
                          filter=TRUE, measure="global", verbose=FALSE) {
  ## No tree
  # Filter
  if (filter) {
    MBC <- learn_MBC(rbind(training_set, validation_set), classes, features)
    performance <- BIC(MBC, rbind(training_set, validation_set))
  }
  # Wrapper
  else {
    MBC <- learn_MBC(training_set, classes, features)
    out <- predict_MBC_dataset_veryfast(MBC, validation_set, classes, features)
    performance <- test_multidimensional(validation_set, out, classes)[[measure]]
  }
  MBCTree <- list("MBC"=MBC, "performance"=performance)
  # Try to improve performance splitting features in the tree
  return(learn_MBCTree_aux(MBCTree, training_set, validation_set, classes, features,
                           N=nrow(training_set)+nrow(validation_set), filter, measure, verbose))
}

###
# Auxiliar method for growing an MBCTree <MBCTree>
# The recursive partitioning is made in this method, until:
#   - There is no significant improvement (in the global/average accuracy for wrapper, or in the BIC for filter)
#   - There is no enough data for learning or validating split MBCs
#   - There is only one feature variable left
###
learn_MBCTree_aux <- function(MBCTree, training_set, validation_set, classes, features, N,
                              filter, measure, verbose) {
  # Don't split if there is only one feature left
  if (length(features) == 1) { return(append(MBCTree, list("leaf"=TRUE))) }
  # If no improvement, this MBC will be a leaf in the tree
  best_performance <- MBCTree$performance
  initial_performance <- MBCTree$performance
  if (verbose) { print(paste0("Initial performance ", initial_performance)) }
  leaf <- TRUE
  # Decide which feature to split on, if there is one that improves
  for (i in 1:length(features)) {
    noData <- FALSE
    feature <- features[i]
    features_rest <- features[-i]
    # Learn an MBC for each label
    MBCs <- list()
    training_set_filtered <- list()
    validation_set_filtered <- list()
    labels <- attributes(MBCTree$MBC[[feature]]$prob)$dimnames[[1]]
    for (j in 1:length(labels)) {
      lab <- labels[j]
      training_set_filtered[[lab]] <- training_set[training_set[,feature] == lab, c(classes,features_rest)]
      validation_set_filtered[[lab]] <- validation_set[validation_set[,feature] == lab, c(classes,features_rest)]
      # Filter
      if (filter) {
        # If we don't have enough data to train and test, don't try this feature
        if (nrow(training_set_filtered[[lab]]) +
            nrow(validation_set_filtered[[lab]]) < 100) {
          noData <- TRUE; break
        }
      }
      # Wrapper
      else {
        # If we don't have enough data to train, don't try this feature
        if (nrow(training_set_filtered[[lab]]) < 90) { noData <- TRUE; break }
        # If we don't have enough data to test, don't try this feature
        if (nrow(validation_set_filtered[[lab]]) < 10) { noData <- TRUE; break } 
      }
    }
    if (noData == TRUE) {
      if (verbose) { print(paste0("Performance ", feature, " unknown")) }
      next
    }
    if (!filter) {
      out <- validation_set[,classes] # To maintain factors format
      true <- out
    }
    index <- 1
    # If filter, penalize growing the tree (BIC)
    if (filter) { performance <- -log(N)/2 * (length(labels)-1) }
    else { performance <- 0 }
    # Score of all the MBCs children
    for (j in 1:length(labels)) {
      lab <- labels[j]
      # Filter
      if (filter) {
        MBCs[[lab]] <- learn_MBC(rbind(training_set_filtered[[lab]],
                                       validation_set_filtered[[lab]]),
                                 classes, features_rest)
        performance <- performance + BIC(MBCs[[lab]], rbind(training_set_filtered[[lab]],
                                                            validation_set_filtered[[lab]])) +
                                     log((nrow(training_set_filtered[[lab]]) + nrow(validation_set_filtered[[lab]])) /
                                         (nrow(training_set) + nrow(validation_set))) *
                                     (nrow(training_set_filtered[[lab]]) + nrow(validation_set_filtered[[lab]]))
      }
      # Wrapper
      else {
        MBCs[[lab]] <- learn_MBC(training_set_filtered[[lab]], classes, features_rest)
        index_prev <- index
        index <- index + nrow(validation_set_filtered[[lab]])
        true[index_prev:(index-1), ] <- validation_set_filtered[[lab]][,classes]
        out[index_prev:(index-1), ] <- predict_MBC_dataset_veryfast(MBCs[[lab]],
                                        validation_set_filtered[[lab]], classes, features_rest)
      }
    }
    if (!filter) { performance <- test_multidimensional(true, out, classes)[[measure]] }
    if (verbose) { print(paste0("Performance ", feature, " ", performance)) }
    # Has it improved? YES:
    if (performance > best_performance) {
      best_MBCs <- MBCs
      best_performance <- performance
      best_feature <- feature
      best_features_rest <- features_rest
      best_training_set_filtered <- training_set_filtered
      best_validation_set_filtered <- validation_set_filtered
      leaf <- FALSE
    }
  }
  # Don't learn noise
  if (!filter & (best_performance - initial_performance) * nrow(validation_set) < 10) {
    leaf <- TRUE 
  }
  # There is no improvement or enough data
  if (leaf == TRUE) {
    if (verbose) { print("This branch is pruned: not enough improvement or data") }
    # Learn with training+validation sets as the recursion ends
    MBCTree$MBC <- learn_MBC(rbind(training_set, validation_set), classes, features)
    return(append(MBCTree, list("leaf"=TRUE)))
  }
  # Else -> Split
  if (verbose) { print(paste0("Recursion continues with ", best_feature)) }
  labels <- attributes(MBCTree$MBC[[best_feature]]$prob)$dimnames[[1]]
  for (i in 1:length(labels)) {
    lab <- labels[i]
    # Performance
    MBC <- best_MBCs[[lab]]
    if (filter) {
      performance <- BIC(MBC, rbind(best_training_set_filtered[[lab]],
                                    best_validation_set_filtered[[lab]]))
    }
    else {
      out <- predict_MBC_dataset_veryfast(MBC, best_validation_set_filtered[[lab]], classes, best_features_rest)
      performance <- test_multidimensional(best_validation_set_filtered[[lab]], out, classes)[[measure]]
    }
    # Grow tree
    MBC_subtree <- list("MBC"=MBC, "performance"=performance)
    best_MBCs[[lab]] <- learn_MBCTree_aux(MBC_subtree, best_training_set_filtered[[lab]],
                                          best_validation_set_filtered[[lab]], classes,
                                          best_features_rest, N, filter, measure, verbose)
  }
  return(append(MBCTree, list("feature"=best_feature, "leaf"=FALSE,
                              "MBC_split"=best_MBCs, "performance_split"=best_performance)))
}

################################################################################################################
##########################                                                            ##########################
##########################                     RANDOM GENERATION                      ##########################
##########################                                                            ##########################
################################################################################################################

###
# <features> and <classes> : character (vector)
# <parents> : numeric
#
# Generates a random MBC structure with feature variables <features> and class variables <classes>
# such that nodes in class and feature subgraphs have at most <parents> parents
#
# Returns the randomly generated MBC structure
#   <random_graph> : bn
###
random_MBC_structure <- function(features, classes, parents) {
  feature_graph <- random.graph(features, method="melancon", max.in.degree=parents)
  class_graph <- random.graph(classes, method="melancon", max.in.degree=parents)
  arcs <- rbind(feature_graph$arcs, class_graph$arcs)
  # Add arcs from features to classes with p=50%
  for (i in 1:length(features)) {
    for (j in 1:length(classes)) {
      if (sample(0:1, 1)) {
        arcs <- rbind(arcs, c(classes[j], features[i]))
      }
    }
  }
  random_graph <- empty.graph(c(features, classes))
  arcs(random_graph) <- arcs
  return(random_graph)
}

###
# <features> and <classes> : character (vector)
# <parents> : numeric
#
# Generates a random MBC with feature variables <features> and class variables <classes>
# such that nodes in class and feature subgraphs have at most <parents> parents.
# Parameters are forced to be extreme, i.e., lower than 0.3 and greater than 0.7
#
# Returns the randomly generated MBC
#   <random_MBC> : c('bn.fit', 'bn.fit.dnet')
###
random_MBC <- function(features, classes, parents) {
  # Random structure
  random_graph <- random_MBC_structure(features, classes, parents)
  # Random parameters for all BINARY nodes
  cpts <- list()
  variables <- c(features, classes)
  for (i in 1:length(variables)) {
    var <- variables[i]
    cpt <- double()
    parents <- random_graph$nodes[[var]]$parents
    nparents <- length(parents)
    for (j in 1:2^nparents) {
      # Probabilities in the extreme
      if (sample(0:1, 1)) {
        random <- runif(1, 0.0, 0.3)
      }
      else {
        random <- runif(1, 0.7, 1.0)
      }
      cpt <- c(cpt, random, 1-random)
    }
    dim(cpt) = rep(2, nparents+1)
    # Make labels be TRUE, FALSE instead of A, B
    if (nparents == 0) {
      dimnames(cpt) <- list(c("TRUE", "FALSE"))
    }
    else {
      labels <- list()
      labels[[var]] <- c("TRUE", "FALSE")
      for (j in 1:nparents) {
        labels[[parents[j]]] <- c("TRUE", "FALSE")
      }
      dimnames(cpt) <- labels
    }
    cpts[[var]] <- cpt
  }
  random_MBC <- custom.fit(random_graph, dist=cpts)
  return(random_MBC)
}

###
# <features> and <classes> : character (vector)
# <depth> and <parents> : numeric
#
# Generates a random MBCTree of depth <depth> with class variables <classes> and
# feature variables <features> such that nodes in class and feature subgraphs
# of all the MBCs leaf in the tree have at most <parents> parents. The tree is
# complete, i.e., all the paths from the root to a leaf have exactly <depth> internal nodes
#
# Returns the randomly generated MBCTree as a recursive list such that:
#  + A leaf node is a list with:
#    - Key <MBC> : c('bn.fit', 'bn.fit.dnet')
#        MBC associated to the leaf node
#    - Key <leaf> : logical
#        A TRUE value meaning this is a leaf node
#  + An internal node is a list with:
#    - Key <feature> : character
#        The feature variable that splits the data in the current internal node
#    - Key <MBC_split> : list
#        The possible values of the feature variable <feature> are the keys of the list
#        Each element is another list representing the sub-MBCTree associated to each child
#    - Key <leaf> : logical
#        A FALSE value meaning this is an internal node
###
random_MBCTree <- function(features, classes, depth, parents) {
  # Leaf
  if (depth == 0) {
    MBC <- random_MBC(features, classes, parents)
    return(list("MBC"=MBC, "leaf"=TRUE))
  }
  # Split
  else {
    feature <- sample(1:length(features), 1)
    features_rest_v <- rep(TRUE, length(features))
    features_rest_v[feature] <- FALSE
    features_rest <- features[features_rest_v]
    MBC_split <- list("TRUE"  = random_MBCTree(features_rest, classes, depth-1, parents),
                      "FALSE" = random_MBCTree(features_rest, classes, depth-1, parents))
    return(list("MBC_split"=MBC_split, "leaf"=FALSE, "feature"=features[feature]))
  }
}

###
# <MBCTree> : Recursive list as explained before
# <size> : numeric
#
# Randomly samples a data set of <size> cases from the MBCTree <MBCTree>.
# For this, a data subset of random size is simulated for each MBC leaf by
# using probabilistic logic sampling. It is imposed that each subset contributes
# at least a fixed percentage to the whole data set (a 20% in a recursive manner)
#
# The MBCTree must have BINARY variables with TRUE and FALSE possible values
#
# Returns a data.frame with the simulated data set
###
sample_MBCTree <- function(MBCTree, size) {
  # Leaf
  if (MBCTree$leaf) {
    return(rbn(MBCTree$MBC, n=size))
  }
  # Split
  else {
    #random <- runif(1, 0.3, 0.7)
    random <- 0.5
    dataset_true <- sample_MBCTree(MBCTree$MBC_split$`TRUE`, round(size*random))
    dataset_true[[MBCTree$feature]] <- as.factor(rep("TRUE", nrow(dataset_true)))
    dataset_false <- sample_MBCTree(MBCTree$MBC_split$`FALSE`, round(size*(1-random)))
    dataset_false[[MBCTree$feature]] <- as.factor(rep("FALSE", nrow(dataset_false)))
    return(rbind(dataset_true,dataset_false))
  }
}

################################################################################################################
##########################                                                            ##########################
##########################                           UTILS                            ##########################
##########################                                                            ##########################
################################################################################################################

###
# <classes> and <features> : character (vector)
#
# Returns all possible arcs of an MBC with class variables <classes> and feature variables <features>
#  <arcs> : matrix
###
MBC_possible_arcs <- function(classes, features) {
  # Possible arcs to add
  size <- length(classes) * (length(classes)-1) + # Class subgraph
    length(features) * (length(features)-1) +     # Feature subgraph
    length(classes) * length(features)            # Bridge subgraph
  arcs <- matrix(nrow=size, ncol=2, dimnames=list(NULL, c("from", "to")))
  index = 1
  for (i in 1:length(classes)) {
    # Class subgraph
    for (j in 1:length(classes)) {
      if (i != j) {
        arcs[index, ] <- c(classes[i], classes[j])
        index <- index + 1
      }
    }
    # Bridge subgraph
    for (j in 1:length(features)) {
      arcs[index, ] <- c(classes[i], features[j])
      index <- index + 1
    }
  }
  # Feature subgraph
  for (i in 1:length(features)) {
    for (j in 1:length(features)) {
      if (i != j) {
        arcs[index, ] <- c(features[i], features[j])
        index <- index + 1
      }
    }
  }
  return(arcs)
}

###
# Returns the internal structure of the MBCTree <MBCTree>
#
# Example:
#   > X3 (root note)
#     > X6 (depth 1)
#       > X2 (depth 2)
#         > X1 (depth 3)
#       > X1 (depth 2)
#       > X8 (depth 2)
#     > X8 (depth 1)
###
info_MBCTree <- function(MBCTree) {
  return(info_MBCTree_aux(MBCTree, 0))
}

info_MBCTree_aux <- function(MBCTree, depth) {
  if (!MBCTree$leaf) {
    str <- paste0(strrep("  ", depth), "> ", MBCTree$feature, "\n")
    for (child in 1:length(MBCTree$MBC_split)) {
      str <- paste0(str, info_MBCTree_aux(MBCTree$MBC_split[[child]], depth+1))
    }
    return(str)
  }
}

} # End function definitions

################################################################################################################
##########################                                                            ##########################
##########################                        EXPERIMENTS                         ##########################
##########################                                                            ##########################
################################################################################################################

compare_models <- function(training_set, test_set, classes, features) {
  base_classifiers <- c("RF", "SVM", "NB")
  base_classifiers <- c("RF", "NB")
  results <- list()
  
  ############  Binary Relevance
  for (i in 1:length(base_classifiers)) {
    t <- proc.time()
    model <- br(training_set, base_classifiers[i])
    t <- proc.time()-t
    pred <- fixed_threshold(predict(model, test_set), 0.5)
    pred <- as.data.frame(as.matrix(pred))
    result <- test_multidimensional(test_set$dataset, pred, classes)
    result$t <- t[[1]]
    results[[paste0("br-", base_classifiers[i])]] <- result
  }
  
  ############ Label Powerset
  for (i in 1:length(base_classifiers)) {
    t <- proc.time()
    model <- lp(training_set, base_classifiers[i])
    t <- proc.time()-t
    pred <- fixed_threshold(predict(model, test_set), 0.5)
    pred <- as.data.frame(as.matrix(pred))
    result <- test_multidimensional(test_set$dataset, pred, classes)
    result$t <- t[[1]]
    results[[paste0("lp-", base_classifiers[i])]] <- result
  }
  
  ############ Classifier Chain
  for (i in 1:length(base_classifiers)) {
    t <- proc.time()
    model <- cc(training_set, base_classifiers[i])
    t <- proc.time()-t
    pred <- fixed_threshold(predict(model, test_set), 0.5)
    pred <- as.data.frame(as.matrix(pred))
    result <- test_multidimensional(test_set$dataset, pred, classes)
    result$t <- t[[1]]
    results[[paste0("cc-", base_classifiers[i])]] <- result
  }
  
  ############ Random k-labelsets (needs classes as numeric)
  training_set_factor <- training_set$dataset
  training_set$dataset[,classes] <- as.data.frame(apply(training_set$dataset[,classes], 2, function(x) as.integer(x)))
  for (i in 1:length(base_classifiers)) {
    t <- proc.time()
    model <- rakel(training_set, base_classifiers[i])
    t <- proc.time()-t
    pred <- fixed_threshold(predict(model, test_set), 0.5)
    pred <- as.data.frame(as.matrix(pred))
    result <- test_multidimensional(test_set$dataset, pred, classes)
    result$t <- t[[1]]
    results[[paste0("rakel-", base_classifiers[i])]] <- result
  }
  
  ############ MBC pure-fitler
  t <- proc.time()
  MBC_filter <- learn_MBC(training_set_factor, classes, features)
  t <- proc.time()-t
  pred <- predict_MBC_dataset_veryfast(MBC_filter, test_set$dataset, classes, features)
  result <- test_multidimensional(test_set$dataset, pred, classes)
  result$t <- t[[1]]
  results[["MBC-filter"]] <- result
  
  ## Split training in training (80) and validation (20)
  train_val_set <- iterative.stratification.holdout(training_set, 80) # needa classes as numeric, they alerady are
  
  # Remove .labelcount and .SCUMBLE columns
  train_val_set$train$dataset <- train_val_set$train$dataset[,variables]
  train_val_set$test$dataset <- train_val_set$test$dataset[,variables]
  train_val_set$train$attributesIndexes <- head(train_val_set$train$attributesIndexes,-2)
  train_val_set$test$attributesIndexes  <- head(train_val_set$test$attributesIndexes,-2)
  for (x in 1:length(classes)) {
    # Maintain factor levels, important if a subset has no data of a specific level
    train_val_set$train$dataset[,classes[x]] <- factor(train_val_set$train$dataset[,classes[x]],
                                                       levels(training_set_factor[,classes[x]]))
    train_val_set$test$dataset[,classes[x]]  <- factor(train_val_set$test$dataset[,classes[x]],
                                                       levels(training_set_factor[,classes[x]]))
  }
  
  ############ MBTree filter
  t <- proc.time()
  MBCTree_filter <- learn_MBCTree(train_val_set$train$dataset, train_val_set$test$dataset,
                                  classes, features, filter=TRUE, verbose=FALSE)
  t <- proc.time()-t
  pred <- predict_MBCTree_dataset_veryfast(MBCTree_filter, test_set$dataset, classes, features)
  result <- test_multidimensional(pred$true, pred$out, classes)
  result$t <- t[[1]]
  result$tree <- info_MBCTree(MBCTree_filter)
  results[["MBCTree-filter"]] <- result
  
  ############ MBTree wrapper global
  t <- proc.time()
  MBCTree_wrapper_global <- learn_MBCTree(train_val_set$train$dataset, train_val_set$test$dataset,
                                          classes, features, filter=FALSE, measure="global", verbose=FALSE)
  t <- proc.time()-t
  pred <- predict_MBCTree_dataset_veryfast(MBCTree_wrapper_global, test_set$dataset, classes, features)
  result <- test_multidimensional(pred$true, pred$out, classes)
  result$t <- t[[1]]
  result$tree <- info_MBCTree(MBCTree_wrapper_global)
  results[["MBCTree-wrapper-global"]] <- result
  
  ############ MBTree wrapper average
  t <- proc.time()
  MBCTree_wrapper_average <- learn_MBCTree(train_val_set$train$dataset, train_val_set$test$dataset,
                                           classes, features, filter=FALSE, measure="average", verbose=FALSE)
  t <- proc.time()-t
  pred <- predict_MBCTree_dataset_veryfast(MBCTree_wrapper_average, test_set$dataset, classes, features)
  result <- test_multidimensional(pred$true, pred$out, classes)
  result$t <- t[[1]]
  result$tree <- info_MBCTree(MBCTree_wrapper_average)
  results[["MBCTree-wrapper-average"]] <- result
  
  return(results)
}

# Show relevant results from the cross-validation procedure
show_results <- function(results) {
  # Compute mean+-std
  models <- names(results[[1]])
  measures <- names(unlist(results[[1]][[models[1]]]))
  
  mean_results <- matrix(rep(0, length(models)*length(measures)),
                         nrow = length(models),
                         ncol = length(measures), byrow = TRUE,
                         dimnames = list(models, measures))
  std_results <- mean_results
  
  # Mean
  `%+=%` = function(e1,e2) eval.parent(substitute(e1 <- e1 + e2))
  for (fold in 1:length(results)) {
    for (model in 1:length(models)) {
      mean_results[models[model],] %+=%
        as.numeric(unlist(results[[fold]][[models[model]]])[measures])
    }
  }
  mean_results <- mean_results/length(results)
  
  # Std
  for (fold in 1:length(results)) {
    for (model in 1:length(models)) {
      std_results[models[model],] %+=%
        (as.numeric(unlist(results[[fold]][[models[model]]])[measures]) - mean_results[model,])^2
    }
  }
  std_results <- sqrt(std_results/length(results))
  
  rel_measures <- c("global", "average", "t")
  mean_std <- matrix(paste(format(round(mean_results[,rel_measures],4), nsmall = 2),
                           format(round( std_results[,rel_measures],4), nsmall = 2), sep=" ± "), 
                     nrow=nrow(mean_results), dimnames=dimnames(mean_results[,rel_measures]))
  
  return(mean_std)
}

# Concatenate results of the k-cv
concatenate_results <- function(results) {
  models <- names(results[[1]])
  measures <- c("global", "average", "t")
  
  concatenate_results <- vector("list", length(measures))
  names(concatenate_results) <- measures
  for (i in 1:length(measures)) {
    # Initialize
    concatenate_results[[i]] <- matrix(rep(0, length(results)*length(models)),
                                       nrow = length(results),
                                       ncol = length(models), byrow = TRUE,
                                       dimnames = list(NULL, models))
    # Fill
    for (j in 1:length(results)) {
      for (k in 1:length(models)) {
        model <- models[k]
        concatenate_results[[i]][j,model] <- results[[j]][[model]][[measures[i]]]
      }
    }
  }

  return(concatenate_results)
}
  
########### SYNTHETIC DATA

m <- 10         # Number of features in the MBCs leaf
d <- 4          # Number of class variables
s <- 2          # Depth of the MBCTree
parents <- 3    # Maximum number of parents of a node in the class and feature subgraphs
N <- 10000      # Size of the simulated data set

# C1, ..., Cd
classes = sapply(1:d, function(x) paste("C", x, sep=""))
# X1, ..., Xm
features = sapply(1:(m+s), function(x) paste("X", x, sep=""))

# Number of experiments
results <- list()
executions <- 1
for (case in 1:executions) {
  print(paste0("Execution ", case))
  # Simulate MBCTree and data set
  MBCTree_init <- random_MBCTree(features, classes, s, parents)
  cat(info_MBCTree(MBCTree_init))
  print("-------")
  dataset <- sample_MBCTree(MBCTree_init, N)
  variables <- names(dataset)

  # k-cross validation
  k <- 10
  dataset[,classes] <- as.data.frame(apply(dataset[,classes], 2, function(x) as.integer(as.logical(x))))
  mldr <- mldr_from_dataframe(dataset, labelIndices = match(classes,names(dataset)))
  folds <- iterative.stratification.kfolds(mldr, k=k)
  
  for (j in 1:k) {
    # Remove .labelcount and .SCUMBLE columns
    folds[[j]]$train$dataset <- folds[[j]]$train$dataset[,variables]
    folds[[j]]$test$dataset <- folds[[j]]$test$dataset[,variables]
    folds[[j]]$train$attributesIndexes <- head(folds[[j]]$train$attributesIndexes,-2)
    folds[[j]]$test$attributesIndexes  <- head(folds[[j]]$test$attributesIndexes,-2)
    for (x in 1:length(classes)) {
      folds[[j]]$train$dataset[,classes[x]] <- factor(folds[[j]]$train$dataset[,classes[x]])
      folds[[j]]$test$dataset[,classes[x]]  <- factor(folds[[j]]$test$dataset[,classes[x]])
    }
  }
  
  # Compare models
  results_case <- foreach(fold = 1:k, .packages=c('utiml','bnlearn','mldr.datasets')) %dopar% {
    compare_models(folds[[fold]]$train, folds[[fold]]$test, classes, features)
  }
  
  results <- c(results, results_case)
  
  # Show tree structures
  for (fold in 1:k) {
    cat(results_case[[fold]][["MBCTree-filter"]][["tree"]])
    cat("-----------------\n")
    cat(results_case[[fold]][["MBCTree-wrapper-global"]][["tree"]])
    cat("-----------------\n")
    cat(results_case[[fold]][["MBCTree-wrapper-average"]][["tree"]])
    cat("=================\n")
  }
}

# Compute mean+-std
print(show_results(results))

# Friedman test
concat_results <- concatenate_results(results)
friedman.test(concat_results$global)
friedman.test(concat_results$average)

p_values <- matrix(rep(0, length(models)*length(models)),
                   nrow = length(models),
                   ncol = length(models), byrow = TRUE,
                   #dimnames = list(models, models))
                   dimnames = list(NULL, NULL))

for (i in 1:(length(models)-1)) {
  for (j in (i+1):length(models)) {
    p_values[i,j] <- wilcox.test(concat_results$global[,i],
                                 concat_results$global[,j],
                                 paired=TRUE, exact=TRUE)$p.value
  }
}

########### BENCHMARK MULTI-LABEL DATA SETS

datasets <- list.files(path = "datasets", full.names = TRUE)

for (i in 1:length(datasets)) {
  mldr <- readRDS(datasets[i])
  
  mldr$dataset <- na.omit(mldr$dataset)
  
  classes <- colnames(mldr$dataset)[mldr$labels$index]
  features <- colnames(mldr$dataset)[mldr$attributesIndexes]
  variables <- c(classes, features)
  
  # Discretize features. Try catch in case they are already discretized
  mldr$dataset[,features] <- as.data.frame(apply(mldr$dataset[,features], 2,
                                                 function(x) tryCatch({
                                                   discretize(x, method='frequency', breaks=3)},
                                                  error=function(e){tryCatch({
                                                   discretize(x, method='cluster', breaks=3)},
                                                  error=function(e){x})})))
  
  # FSS
  mldr_factor <- as.data.frame(apply(mldr$dataset[,variables], 2, function(x) as.factor(x)))
  desired_features <- 100
  if (length(features) > desired_features) {
    weights <- data.frame("attr_importance"=rep(0, length(features)))
    rownames(weights) <- features
    for (j in 1:length(classes)) {
      class <- classes[j]
      weights <- pmax(weights,
                      information.gain(as.simple.formula(".", class),
                                       mldr_factor[,c(features,class)]))
    }
    rel_features <- cutoff.k(weights, desired_features)
  }
  else {
    rel_features <- features
  }
  
  # Remove irrelevant features
  variables <- c(classes, rel_features)
  mldr <- mldr_from_dataframe(mldr$dataset[,variables], 1:length(classes), name=mldr$name)
  
  # k-cross validation
  k <- 10
  folds <- iterative.stratification.kfolds(mldr, k=k)
 
  for (j in 1:k) {
    # Remove .labelcount and .SCUMBLE columns
    folds[[j]]$train$dataset <- folds[[j]]$train$dataset[,variables]
    folds[[j]]$test$dataset <- folds[[j]]$test$dataset[,variables]
    folds[[j]]$train$attributesIndexes <- head(folds[[j]]$train$attributesIndexes,-2)
    folds[[j]]$test$attributesIndexes  <- head(folds[[j]]$test$attributesIndexes,-2)
    for (x in 1:length(classes)) {
      # Maintain factor levels, important if a subset has no data of a specific level
      folds[[j]]$train$dataset[,classes[x]] <- factor(folds[[j]]$train$dataset[,classes[x]],
                                                      levels(mldr_factor[,classes[x]]))
      folds[[j]]$test$dataset[,classes[x]]  <- factor(folds[[j]]$test$dataset[,classes[x]],
                                                      levels(mldr_factor[,classes[x]]))
    }
  }

  # Compare models
  results <- foreach(fold = 1:k, .packages=c('utiml','bnlearn','mldr.datasets')) %dopar% {
    compare_models(folds[[fold]]$train, folds[[fold]]$test, classes, rel_features)
  }
  
  # Save results
  sink(paste0(mldr$name,".txt"))
  
  print(paste0("Features: ", length(features)))
  print(paste0("Relevants: ", length(rel_features)))
  
  show_results(results)
  
  sink()
}
