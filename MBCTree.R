#***************************************************************************************************************
#***************************************************************************************************************
#
# Authors: Santiago Gil-Begue, Pedro Larrañaga and Concha Bielza
#
# Notes: This code is part of the publication with name 'Multi-dimensional Bayesian network classifier trees'
#        published in Proceedings of the 19th International Conference on Intelligent Data Engineering and
#        Automated Learning, Lecture Notes in Artificial Intelligence, Springer (2018).
#
# Further information: sgil@fi.upm.es
#
#***************************************************************************************************************
#***************************************************************************************************************

#install.packages("bnlearn")
library("bnlearn")
#source("http://bioconductor.org/biocLite.R")
#biocLite(c("graph", "RBGL", "Rgraphviz"))
#install.packages("gRain", dependencies=TRUE)
library("gRain")
#install.packages("utiml")
library("utiml")

################################################################################################################
##########################                                                            ##########################
##########################              PERFORMANCE EVALUATION MEASURES               ##########################
##########################                                                            ##########################
################################################################################################################

##
# <test_set> and <out> : data.frame
# <clases> : character (vector)
# 
# Calculates the multi-label performance over the binary class variables <clases>,
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
test_multilabel <- function(test_set, out, clases) {
  # Convert from data.frames to <utiml> required objects (mldr and mlresult)
  true <- as.data.frame(sapply(test_set[, clases], as.logical)) * 1
  true_mldr <- mldr_from_dataframe(true, labelIndices = 1:length(clases))
  out <- as.data.frame(sapply(out, as.logical)) * 1
  out_mlresult <- as.mlresult(out, probability = TRUE)
  # Confusion matrices
  mlconfmat <- multilabel_confusion_matrix(true_mldr, out_mlresult)
  # Multi-label performance evaluation measures. All possible are given
  measures <- multilabel_evaluate(mlconfmat, measures = c("all"))
  return(list("mlconfmat" = mlconfmat, "measures" = measures))
}

###
# <test_set> and <out> : data.frame
# <clases> : character (vector)
# 
# Calculates the multi-dimensional performance over the class variables <clases>,
# such that <test_set> are the true values of the variables and <out> its predictions
#
# Prints useful information: per-class accuracy, average accuracy and global accuracy
#
# Returns the global accuracy over the d-dimensional class variable
#   <global_accuracy> : numeric
###
test_multidimensional <- function(test_set, out, clases) {
  # Step needed before '==' operator
  true <- as.data.frame(sapply(test_set[, clases], as.character), stringsAsFactors=FALSE)
  out <- as.data.frame(sapply(out, as.character), stringsAsFactors=FALSE)
  match <- true == out
  exact_match <- rep(TRUE, nrow(test_set))
  classes_accuracy <- numeric()
  for (i in 1:length(clases)) {
    print(clases[i])
    accuracy_i <- mean(match[,i])
    print(accuracy_i)
    classes_accuracy <- c(classes_accuracy, accuracy_i)
    exact_match <- exact_match & match[,i]
  }
  # Average accuracy
  print("Average")
  average_accuracy <- mean(classes_accuracy)
  print(average_accuracy)
  # Global accuracy
  print("Exact")
  global_accuracy <- mean(exact_match)
  print(global_accuracy)
  return(global_accuracy)
}

################################################################################################################
##########################                                                            ##########################
##########################                    PREDICT WITH MODELS                     ##########################
##########################                                                            ##########################
################################################################################################################

###
# <MBC> : CPTgrain grain
# <case> : data.frame (with just one row)
# <clases> and <predictoras> : character (vector)
#
# Predicts the class variables <clases> given the values of the features variables <predictoras>
# of the instance <case> by using the multi-dimensional Bayesian network classifier <MBC>
#
# Returns the most probable explanation (MPE) of the class variables
#   <out> : character (vector)
###
predict_MBC_case <- function(MBC, case, clases, predictoras) {
  net_ev <- gRain::setEvidence(MBC, evidence = lapply(case[predictoras], function(x) as.character(x)))
  res <- gRain::querygrain(net_ev, nodes=clases, type="joint")
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
# <out> : data.frame (nrow(out) == nrow(test_set))
###
predict_MBC_dataset <- function(MBC, test_set, clases, predictoras) {
  out <- data.frame(matrix(ncol = length(clases), nrow = nrow(test_set), dimnames = list(NULL, clases)))
  for (i in 1:nrow(test_set)) {
    # <foo> classes may be given in different order than those in <out>
    foo <- predict_MBC_case(MBC, test_set[i,], clases, predictoras)
    for (j in 1:length(clases)) {
      clase <- clases[j]
      out[i,clase] <- foo[clase]
    }
  }
  return(out)
}

###
# The same as <predict_MBC_case> but using an MBCTree <MBCTree> as the classifier
###
predict_MBCTree_case <- function(MBCTree, case, clases, predictoras) {
  # Reach the corresponding MBC leaf
  MBCTree_aux <- MBCTree
  predictoras_used <- list()
  while (MBCTree_aux$leaf == FALSE) {
    predictoras_used <- c(predictoras_used, MBCTree_aux$pred)
    lab <- case[,MBCTree_aux$pred]
    MBCTree_aux <- MBCTree_aux$MBC_split[[lab]]
  }
  # Predict the case with the reached MBC leaf
  predictoras_rest <- predictoras[!predictoras %in% predictoras_used]
  out <- predict_MBC_case(MBCTree_aux$MBC, case, clases, predictoras_rest)
  return(out)
}

###
# The same as <predict_MBC_dataset> but using an MBCTree <MBCTree> as the classifier
###
predict_MBCTree_dataset <- function(MBCTree, test_set, clases, predictoras) {
  out <- data.frame(matrix(ncol = length(clases), nrow = nrow(test_set), dimnames = list(NULL, clases)))
  for (i in 1:nrow(test_set)) {
    # <foo> classes may be given in different order than those in <out>
    foo <- predict_MBCTree_case(MBCTree, test_set[i,], clases, predictoras)
    for (j in 1:length(clases)) {
      clase <- clases[j]
      out[i,clase] <- foo[clase]
    }
  }
  return(out)
}

################################################################################################################
##########################                                                            ##########################
##########################                        LEARN MODELS                        ##########################
##########################                                                            ##########################
################################################################################################################

###
# <training_set> : data.frame
# <clases> and <predictoras> : character (vector)
#
# Learns an MBC from <training_set> data set with class variables <clases> and feature variables <predictoras>
# in a filter way using hill climbing algorithm and maximizing BIC score.
# Bayesian method is used for the parameter estimation, Laplace rule is used for regularization.
#
# Returns the MBC learned
#  <MBC> : CPTgrain grain
###
learn_MBC <- function(training_set, clases, predictoras) {
  # Black list of arcs from features to classes
  bl <- matrix(nrow = length(clases)*length(predictoras), ncol = 2, dimnames = list(NULL, c("from","to")))
  bl[,"from"] <- rep(predictoras, each = length(clases))
  bl[,"to"] <- rep(clases, length(predictoras))
  # Learn MBC structure
  net <- hc(training_set, blacklist = bl)
  # Fit CPTs
  fitted <- bn.fit(net, training_set, method = "bayes", iss = 1) # iss = 1 -> Laplace
  MBC <- as.grain(fitted)
  return(MBC)
}

###
# <training_set> and <validation_set> : data.frame
# <clases> and <predictoras> : character (vector)
#
# Learns an MBCTree from <training_set> and <validation_set> data sets with class
# variables <clases> and feature variables <predictoras> as follows:
#  - <training_set> is used to learn MBCs
#  - <validation_set> is used to evaluate the MBCs so that the best split can be computed
# The algorithm follows a wrapper approach guided by the global accuracy
#
# Returns the learned MBCTree as a recursive list such that:
#  + A leaf node is a list with:
#    - Key <MBC> : CPTgrain grain
#        MBC associated to the leaf node
#    - Key <leaf> : logical
#        A TRUE value meaning this is a leaf node
#    - Key <performance> : numeric
#        Global accuracy over the corresponding portion of <validation_set> of the MBC
#  + An internal node is a list with:
#    - Key <pred> : character
#        The feature variable that splits the data in the current internal node
#    - Key <MBC_split> : list
#        The possible values of the feature variable <pred> are the keys of the list
#        Each element is another list representing the sub-MBCTree associated to each child
#    - Key <leaf> : logical
#        A FALSE value meaning this is an internal node
#    - Key <performance> : numeric
#        Global accuracy over the corresponding portion of <validation_set> of an MBC
#        that would have been placed insted of this internal node 
#    - Key <performance_split> : numeric
#        Global accuracy over the corresponding portion of <validation_set> of the split MBCs
#        on the best feature variable (performance_split > performance)
###
learn_MBCTree <- function(training_set, validation_set, clases, predictoras) {
  # No tree
  MBC <- learn_MBC(training_set, clases, predictoras)
  out <- predict_MBC_dataset(MBC, validation_set, clases, predictoras)
  performance <- test_multidimensional(validation_set, out, clases)
  MBCTree <- list("MBC" = MBC, "performance" = performance)
  # Try to improve accuraccy splitting predictoras in the tree
  return(learn_MBCTree_aux(MBCTree, training_set, validation_set, clases, predictoras))
}

###
# Auxiliar method for growing an MBCTree <MBCTree>
# The recursive partitioning is made in this method, until:
#   - There is no significant improvent in the global accuracy
#   - There is no enough data for learning or validating split MBCs
#   - There is only one feature variable left
###
learn_MBCTree_aux <- function(MBCTree, training_set, validation_set, clases, predictoras) {
  # Don't split if there is only one predictora left
  if (length(predictoras) == 1) { return(append(MBCTree, list("leaf" = TRUE))) }
  # If no improvement, this MBC will be a leaf in the tree
  best_accuracy <- MBCTree$performance
  initial_accuracy <- MBCTree$performance
  print(paste0("Accuracy inicial ", best_accuracy))
  leaf <- TRUE
  # Decide which predictora to split, if there is one that improves the accuracy
  for (i in 1:length(predictoras)) {
    noData <- FALSE
    pred <- predictoras[i]
    predictoras_rest <- predictoras[!predictoras %in% pred]
    # Create an MBC for each label
    MBCs <- list()
    training_set_filtered <- list()
    validation_set_filtered <- list()
    labels <- MBCTree$MBC$universe$levels[pred][[1]]
    for (j in 1:length(labels)) {
      lab <- labels[j]
      # If we don't have enough data to train, don't try this predictora
      training_set_filtered[[lab]] <- training_set[training_set[,pred] == lab, c(clases,predictoras_rest)]
      if (nrow(training_set_filtered[[lab]]) < 100) { noData <- TRUE; break }
      # If we don't have enough data to test, don't try this predictora
      validation_set_filtered[[lab]] <- validation_set[validation_set[,pred] == lab, c(clases,predictoras_rest)]
      if (nrow(validation_set_filtered[[lab]]) < 50) { noData <- TRUE; break }
      # Else, try it
      MBCs[[lab]] <- learn_MBC(training_set_filtered[[lab]], clases, predictoras_rest)
    }
    if (noData == TRUE) { print(paste0("Accuracy ", pred, " unknown")); next }
    # Predict test set
    out <- data.frame(matrix(ncol = length(clases),nrow = nrow(validation_set), dimnames = list(NULL, clases)))
    for (j in 1:nrow(validation_set)) {
      lab <- validation_set[j,pred]
      foo <- predict_MBC_case(MBCs[[lab]], validation_set[j,], clases, predictoras_rest)
      for (k in 1:length(clases)) {
        clase <- clases[k]
        out[j,clase] <- foo[clase]
      }
    }
    # Performance
    performance <- test_multidimensional(validation_set, out, clases)
    accuracy <- performance
    print(paste0("Accuracy ", pred, " ", accuracy))
    # Has it improved? YES:
    if (accuracy > best_accuracy) {
      best_accuracy <- accuracy
      best_MBCs <- MBCs
      best_performance <- performance
      best_pred <- pred
      best_predictoras_rest <- predictoras_rest
      best_training_set_filtered <- training_set_filtered
      best_validation_set_filtered <- validation_set_filtered
      leaf <- FALSE
    }
  }
  # Don't learn noise
  if ((best_accuracy - initial_accuracy) * nrow(validation_set) < 5) {
    leaf <- TRUE 
  }
  # There is no improvement or enough data
  if (leaf == TRUE) {
    print("Arbol cortado: no hay mejora o no suficiente data")
    return(append(MBCTree, list("leaf" = TRUE)))
  }
  # Else -> Split
  print(paste0("El arbol sigue ", best_pred))
  labels <- MBCTree$MBC$universe$levels[best_pred][[1]]
  for (i in 1:length(labels)) {
    lab <- labels[i]
    # Performance
    MBC <- best_MBCs[[lab]]
    out <- predict_MBC_dataset(MBC, best_validation_set_filtered[[lab]], clases, best_predictoras_rest)
    performance <- test_multidimensional(best_validation_set_filtered[[lab]], out, clases)
    # Grow tree
    MBC_subtree <- list("MBC" = MBC, "performance" = performance)
    best_MBCs[[lab]] <- learn_MBCTree_aux(MBC_subtree, best_training_set_filtered[[lab]],
                                          best_validation_set_filtered[[lab]], clases, best_predictoras_rest)
  }
  return(append(MBCTree, list("pred" = best_pred, "leaf" = FALSE,
                              "MBC_split" = best_MBCs, "performance_split" = best_performance)))
}

################################################################################################################
##########################                                                            ##########################
##########################                     RANDOM GENERATION                      ##########################
##########################                                                            ##########################
################################################################################################################

###
# <predictoras> and <clases> : character (vector)
# <parents> : numeric
#
# Generates a random MBC with feature variables <predictoras> and class variables <classes>
# such that nodes in class and feature subgraphs have at most <parents> parents.
# Parameters are forced to be extreme, i.e., lower than 0.3 and greater than 0.7
#
# Returns the randomly generated MBC
#   <random_MBC> : CPTgrain grain
###
random_MBC <- function(predictoras, clases, parents) {
  ## RANDOM GRAPH
  pred_graph <- random.graph(predictoras, method = "melancon", max.in.degree = parents)
  class_graph <- random.graph(clases, method = "melancon", max.in.degree = parents)
  arcs <- rbind(pred_graph$arcs, class_graph$arcs)
  # Add arcs from features to classes with p=50%
  for (i in 1:length(predictoras)) {
    for (j in 1:length(clases)) {
      if (sample(0:1, 1)) {
        arcs <- rbind(arcs, c(clases[j], predictoras[i]))
      }
    }
  }
  variables <- c(predictoras, clases)
  random_graph = empty.graph(variables)
  arcs(random_graph) <- arcs
  plot(random_graph)
  ## RANDOM PARAMETERS for all BINARY nodes
  cpts <- list()
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
  random_graph_fit <- custom.fit(random_graph, dist = cpts)
  random_MBC <- as.grain(random_graph_fit)
  return(random_MBC)
}

###
# <predictoras> and <clases> : character (vector)
# <depth> and <parents> : numeric
#
# Generates a random MBCTree of depth <depth> with class variables <clases> and
# feature variables <predictoras> such that nodes in class and feature subgraphs
# of all the MBCs leaf in the tree have at most <parents> parents. The tree is
# complete, i.e., all the paths from the root to a leaf have exactly <depth> internal nodes
#
# Returns the randomly generated MBCTree as a recursive list such that:
#  + A leaf node is a list with:
#    - Key <MBC> : CPTgrain grain
#        MBC associated to the leaf node
#    - Key <leaf> : logical
#        A TRUE value meaning this is a leaf node
#  + An internal node is a list with:
#    - Key <pred> : character
#        The feature variable that splits the data in the current internal node
#    - Key <MBC_split> : list
#        The possible values of the feature variable <pred> are the keys of the list
#        Each element is another list representing the sub-MBCTree associated to each child
#    - Key <leaf> : logical
#        A FALSE value meaning this is an internal node
###
random_MBCTree <- function(predictoras, clases, depth, parents) {
  # Leaf
  if (depth == 0) {
    MBC <- random_MBC(predictoras, clases, parents)
    return(list("MBC" = MBC, "leaf" = TRUE))
  }
  # Split
  else {
    pred <- sample(1:length(predictoras), 1)
    predictoras_rest_v <- rep(TRUE, length(predictoras))
    predictoras_rest_v[pred] <- FALSE
    predictoras_rest <- predictoras[predictoras_rest_v]
    MBC_split <- list("TRUE"  = random_MBCTree(predictoras_rest, clases, depth-1, parents),
                      "FALSE" = random_MBCTree(predictoras_rest, clases, depth-1, parents))
    return(list("MBC_split" = MBC_split, "leaf" = FALSE, "pred" = predictoras[pred]))
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
    return(rbn(as.bn.fit(MBCTree$MBC), n = size))
  }
  # Split
  else {
    random <- runif(1, 0.2, 0.8)
    dataset_true <- sample_MBCTree(MBCTree$MBC_split$`TRUE`, round(size*random))
    dataset_true[[MBCTree$pred]] <- as.factor(rep("TRUE", nrow(dataset_true)))
    dataset_false <- sample_MBCTree(MBCTree$MBC_split$`FALSE`, round(size*(1-random)))
    dataset_false[[MBCTree$pred]] <- as.factor(rep("FALSE", nrow(dataset_false)))
    return(rbind(dataset_true,dataset_false))
  }
}

################################################################################################################
##########################                                                            ##########################
##########################                           UTILS                            ##########################
##########################                                                            ##########################
################################################################################################################

###
# Prints the internal structure of the MBCTree <MBCTree>
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
  info_MBCTree_aux(MBCTree, 0)
  invisible()
}

info_MBCTree_aux <- function(MBCTree, depth) {
  if (!MBCTree$leaf) {
    print(paste0(strrep("  ", depth), "> ", MBCTree$pred))
    for (child in 1:length(MBCTree$MBC_split)) {
      info_MBCTree_aux(MBCTree$MBC_split[[child]], depth+1)
    }
  }
}

################################################################################################################
##########################                                                            ##########################
##########################                        EXPERIMENTS                         ##########################
##########################                                                            ##########################
################################################################################################################

m <- 10         # Number of features in the MBCs leaf
d <- 4          # Number of class variables
s <- 2          # Depth of the MBCTree
parents <- 3    # Maximum number of parents of a node in the class and feature subgraphs
N <- 100000     # Size of the simulated data set

# C1, ..., Cd
clases = sapply(1:d, function(x) paste("C", x, sep=""))
# X1, ..., Xm
predictoras = sapply(1:(m+s), function(x) paste("X", x, sep=""))

# Number of experiments
executions <- 1
for (case in 1:executions) {
  print(paste0("Execution ", case))
  # Simulate MBCTree and data set
  MBCTree_init <- random_MBCTree(predictoras, clases, s, parents)
  info_MBCTree(MBCTree_init)
  print("-------")
  dataset <- sample_MBCTree(MBCTree_init, N)

  # Split dataset
  Nreal <- nrow(dataset) # It may not be equal to N because of how sample_MBCTree is implemented
  dataset <- dataset[sample(Nreal), ] # Shuffle
  training <- 0.6
  test <- 0.2
  #validation <- 0.2
  training_set <- dataset[1:(training*Nreal), ]
  test_set <- dataset[(training*Nreal+1):((training+test)*Nreal), ]
  validation_set <- dataset[(((training+test)*Nreal)+1):Nreal, ]
  
  ## Performance of MBC
  # Learn network and parameters
  MBC <- learn_MBC(rbind(training_set, validation_set), clases, predictoras)
  # Predict
  out <- predict_MBC_dataset(MBC, test_set, clases, predictoras)
  # Performance
  print("MBC")
  performance_MBC <- test_multidimensional(test_set, out, clases)
  print("-------")
  
  ## Performance of MBCTree
  # Learn MBCTree
  MBCTree <- learn_MBCTree(training_set, validation_set, clases, predictoras)
  print("-------")
  info_MBCTree(MBCTree)
  print("-------")
  # Predict
  out <- predict_MBCTree_dataset(MBCTree, test_set, clases, predictoras)
  # Performance
  print("MBCTree")
  performance_MBCTree <- test_multidimensional(test_set, out, clases)
  print("========")
}
