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
  true <- as.data.frame(sapply(test_set[, classes], as.logical)) * 1
  true_mldr <- mldr_from_dataframe(true, labelIndices = 1:length(classes))
  out <- as.data.frame(sapply(out, as.logical)) * 1
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
# <out> : data.frame (nrow(out) == nrow(test_set))
###
predict_MBC_dataset <- function(MBC, test_set, classes, features) {
  out <- data.frame(matrix(ncol=length(classes), nrow=nrow(test_set), dimnames=list(NULL, classes)))
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
  options(warn = -1) # Supress warnings
  # Joint class configurations
  classes_levels <- lapply(classes, function(x) attributes(MBC[[x]]$prob)$dimnames[[1]])
  names(classes_levels) <- classes
  classes_joint <- expand.grid(classes_levels)
  I <- nrow(classes_joint)
  # All instances + joint class configurations
  big_matrix_MPE <- cbind(test_set[rep(1:nrow(test_set), each = I), features],
                          classes_joint[rep(seq_len(I), nrow(test_set)),])
  # Obtain MPE as argmax p(classes, features), what is the same as argmax p(classes | features)
  indexes_MPE <- sapply(0:(nrow(test_set)-1),
                        function(x) which.max(logLik(MBC, big_matrix_MPE[(x*I+1):(x*I+I),], by.sample=TRUE)))
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
  out <- predict_MBC_case(MBCTree_aux$MBC, case, classes, features_rest)
  return(out)
}

###
# The same as <predict_MBC_dataset> but using an MBCTree <MBCTree> as the classifier
###
predict_MBCTree_dataset <- function(MBCTree, test_set, classes, features) {
  out <- data.frame(matrix(ncol=length(classes), nrow=nrow(test_set), dimnames=list(NULL, classes)))
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
# in a filter way using hill climbing algorithm and maximizing BIC score.
# Bayesian method is used for the parameter estimation, Laplace rule is used for regularization.
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

###
# <training_set> and <validation_set> : data.frame
# <classes> and <features> : character (vector)
#
# Learns an MBC from <training_set> and <validation_set> data sets with class variables <classes>
# and feature variables <features>. A greedy wrapper strategy is applied, such that it starts from
# a fixed graph, and tries to iteratively add/remove/reverse an arc that improves the global accuracy.
# <training_set> is used for training the current MBC and <validation_set> to check if an accuracy
# improvement has been achieved with the addition or removal of the arc. It stops when no arc
# can be added or deleted such that an improvement is achieved.
# Bayesian method is used for the parameter estimation, Laplace rule is used for regularization.
#
# Returns the learned MBC
#  <MBC_fit_best> : c('bn.fit', 'bn.fit.dnet')
###
learn_MBC_wrapper <- function(training_set, validation_set, classes, features, verbose=FALSE) {
  # Start from a fixed graph (instead from an empty one) in order to avoid early local optima
  MBC_fit <- learn_MBC(training_set, classes, features)
  MBC <- bn.net(MBC_fit)
  # Test it
  out <- predict_MBC_dataset_veryfast(MBC_fit, validation_set, classes, features)
  performance_best <- test_multidimensional(validation_set, out, classes)$global
  MBC_best <- MBC
  MBC_fit_best <- MBC_fit
  # Add/delete/reverse arcs until no one improves
  candidates <- MBC_possible_arcs(classes, features)
  candidates <- candidates[sample(nrow(candidates)),] # Shuffle to get them randomly in an efficient way
  index = nrow(candidates)
  while (index > 0) {
    if (verbose) {
      plot(MBC_best)
      print(paste0("> ", index, " arcs left"))
    }
    # Next arc
    arc <- candidates[index,]; index <- index - 1
    if (verbose) { print(arc) }
    # Check if the arc is not in the Markov Blanket of any class variable
    if (arc["from"] %in% features) {
      interest <- FALSE
      for (i in 1:length(classes)) {
        if (arc["to"] %in% MBC_best$nodes[[classes[[i]]]]$children) {
          interest <- TRUE
          break
        }
      }
      if (!interest) {
        if (verbose) { print("Arc not in the Markov Blanket") }
        next
      }
    }
    # Check if the arc is already present
    arc_present <- FALSE
    arcs <- arcs(MBC_best)
    if (nrow(arcs) > 0) {
      for (i in 1:nrow(arcs)) {
        if (arcs[i,"from"] == arc["from"] & arcs[i,"to"] == arc["to"]) {
          arc_present <- TRUE
          break
        }
      }
    }
    # Drop or reverse arc if present
    drop <- arc_present
    if (arc_present) {
      if (verbose) { print("Dropping or reversing") }
      # Drop arc
      MBC <- drop.arc(MBC_best, from=arc["from"], to=arc["to"])
      MBC_fit <- bn.fit(MBC, training_set, method="bayes", iss=1)
      out <- predict_MBC_dataset_veryfast(MBC_fit, validation_set, classes, features)
      performance <- test_multidimensional(validation_set, out, classes)$global
      # Reverse arc. Don't reverse class-to-feature arcs
      MBC_reverse <- set.arc(MBC, from=arc["to"], to=arc["from"], check.cycles=FALSE)
      performance_reverse <- 0
      if (acyclic(MBC_reverse, directed=TRUE) & !(arc["from"] %in% classes & arc["to"] %in% features)) {
        MBC_fit_reverse <- bn.fit(MBC_reverse, training_set, method="bayes", iss=1)
        out <- predict_MBC_dataset_veryfast(MBC_fit_reverse, validation_set, classes, features)
        performance_reverse <- test_multidimensional(validation_set, out, classes)$global
      }
    }
    # Add arc or its reverse if it is not present
    else {
      if (verbose) { print("Adding arc or its reverse") }
      # Add the arc
      MBC <- set.arc(MBC_best, from=arc["from"], to=arc["to"], check.cycles=FALSE)
      performance <- 0
      # Add the reverse arc
      MBC_reverse <- set.arc(MBC_best, from=arc["to"], to=arc["from"], check.cycles=FALSE)
      performance_reverse <- 0
      # Don't allow cycles
      if (!acyclic(MBC, directed=TRUE) & !acyclic(MBC_reverse, directed=TRUE)) {
        if (verbose) { print("Adding this arc would involve in a cycle") }
        next
      }
      # Test MBC with normal arc
      if (acyclic(MBC, directed=TRUE)) {
        MBC_fit <- bn.fit(MBC, training_set, method="bayes", iss=1)
        out <- predict_MBC_dataset_veryfast(MBC_fit, validation_set, classes, features)
        performance <- test_multidimensional(validation_set, out, classes)$global
      }
      # Test MBC with reverse arc
      if (acyclic(MBC_reverse, directed=TRUE) & !(arc["from"] %in% classes & arc["to"] %in% features)) {
        MBC_fit_reverse <- bn.fit(MBC_reverse, training_set, method="bayes", iss=1)
        out <- predict_MBC_dataset_veryfast(MBC_fit_reverse, validation_set, classes, features)
        performance_reverse <- test_multidimensional(validation_set, out, classes)$global
      }
    }
    # Reverse if it improves
    if (performance_reverse > performance) {
      drop <- FALSE
      if (verbose) { print("Reverting the arc") }
      performance <- performance_reverse
      MBC <- MBC_reverse
      MBC_fit <- MBC_fit_reverse
    }
    if (verbose) {
      print(paste0("Best accuracy: ", performance_best))
      print(paste0("Current accuracy: ", performance))
    }
    # Improves? Yes
    if ((drop & performance >= performance_best) | (!arc_present & performance > performance_best)) {
      if (verbose) { print("Improves") }
      performance_best <- performance
      MBC_best <- MBC
      MBC_fit_best <- MBC_fit
      # Explore all arcs
      candidates <- candidates[sample(nrow(candidates)),]
      index = nrow(candidates)
    }
    # Improves? No
    else {
      if (verbose) { print("Does not improve") }
    }
  }
  return(MBC_fit_best)
}

###
# <training_set> and <validation_set> : data.frame
# <classes> and <features> : character (vector)
#
# Learns an MBCTree from <training_set> and <validation_set> data sets with class
# variables <classes> and feature variables <features> as follows:
#  - <training_set> is used to learn MBCs
#  - <validation_set> is used to evaluate the MBCs so that the best split can be computed
# The algorithm follows a wrapper approach guided by the global accuracy
#
# Returns the learned MBCTree as a recursive list such that:
#  + A leaf node is a list with:
#    - Key <MBC> : c('bn.fit', 'bn.fit.dnet')
#        MBC associated to the leaf node
#    - Key <leaf> : logical
#        A TRUE value meaning this is a leaf node
#    - Key <performance> : numeric
#        Global accuracy over the corresponding portion of <validation_set> of the MBC
#  + An internal node is a list with:
#    - Key <feature> : character
#        The feature variable that splits the data in the current internal node
#    - Key <MBC_split> : list
#        The possible values of the feature variable <feature> are the keys of the list
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
learn_MBCTree <- function(training_set, validation_set, classes, features, verbose=FALSE) {
  # No tree
  MBC <- learn_MBC(training_set, classes, features)
  out <- predict_MBC_dataset_veryfast(MBC, validation_set, classes, features)
  performance <- test_multidimensional(validation_set, out, classes)$global
  MBCTree <- list("MBC"=MBC, "performance"=performance)
  # Try to improve accuraccy splitting features in the tree
  return(learn_MBCTree_aux(MBCTree, training_set, validation_set, classes, features, verbose))
}

###
# Auxiliar method for growing an MBCTree <MBCTree>
# The recursive partitioning is made in this method, until:
#   - There is no significant improvent in the global accuracy
#   - There is no enough data for learning or validating split MBCs
#   - There is only one feature variable left
###
learn_MBCTree_aux <- function(MBCTree, training_set, validation_set, classes, features, verbose) {
  # Don't split if there is only one feature left
  if (length(features) == 1) { return(append(MBCTree, list("leaf"=TRUE))) }
  # If no improvement, this MBC will be a leaf in the tree
  best_performance <- MBCTree$performance
  initial_performance <- MBCTree$performance
  if (verbose) { print(paste0("Initial accuracy ", initial_performance)) }
  leaf <- TRUE
  # Decide which feature to split, if there is one that improves the accuracy
  for (i in 1:length(features)) {
    noData <- FALSE
    feature <- features[i]
    features_rest <- features[-i]
    # Create an MBC for each label
    MBCs <- list()
    training_set_filtered <- list()
    validation_set_filtered <- list()
    labels <- attributes(MBCTree$MBC[[feature]]$prob)$dimnames[[1]]
    for (j in 1:length(labels)) {
      lab <- labels[j]
      # If we don't have enough data to train, don't try this feature
      training_set_filtered[[lab]] <- training_set[training_set[,feature] == lab, c(classes,features_rest)]
      if (nrow(training_set_filtered[[lab]]) < 100) { noData <- TRUE; break }
      # If we don't have enough data to test, don't try this feature
      validation_set_filtered[[lab]] <- validation_set[validation_set[,feature] == lab, c(classes,features_rest)]
      if (nrow(validation_set_filtered[[lab]]) < 50) { noData <- TRUE; break }
      # Else, try it
      MBCs[[lab]] <- learn_MBC(training_set_filtered[[lab]], classes, features_rest)
    }
    if (noData == TRUE) {
      if (verbose) { print(paste0("Accuracy ", feature, " unknown")) }
      next
    }
    # Predict test set
    out <- data.frame(matrix(ncol=length(classes), nrow=nrow(validation_set), dimnames=list(NULL, classes)))
    true <- out
    index <- 1
    for (j in 1:length(labels)) {
      lab <- labels[j]
      index_prev <- index
      index <- index + nrow(validation_set_filtered[[lab]])
      true[index_prev:(index-1), ] <- validation_set_filtered[[lab]][,classes]
      out[index_prev:(index-1), ] <- predict_MBC_dataset_veryfast(MBCs[[lab]],
                                      validation_set_filtered[[lab]], classes, features_rest)
    }
    # Performance
    performance <- test_multidimensional(true, out, classes)$global
    if (verbose) { print(paste0("Accuracy ", feature, " ", performance)) }
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
  if ((best_performance - initial_performance) * nrow(validation_set) < 5) {
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
    out <- predict_MBC_dataset_veryfast(MBC, best_validation_set_filtered[[lab]], classes, best_features_rest)
    performance <- test_multidimensional(best_validation_set_filtered[[lab]], out, classes)$global
    # Grow tree
    MBC_subtree <- list("MBC"=MBC, "performance"=performance)
    best_MBCs[[lab]] <- learn_MBCTree_aux(MBC_subtree, best_training_set_filtered[[lab]],
                                          best_validation_set_filtered[[lab]], classes,
                                          best_features_rest, verbose)
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
    random <- runif(1, 0.2, 0.8)
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
  size <- length(classes) * (length(classes)-1) +   # Class subgraph
    length(features) * (length(features)-1) + # Feature subgraph
    length(classes) * length(features)        # Bridge subgraph
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
    print(paste0(strrep("  ", depth), "> ", MBCTree$feature))
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

m <- 11         # Number of features in the MBCs leaf
d <- 4          # Number of class variables
s <- 1          # Depth of the MBCTree
parents <- 3    # Maximum number of parents of a node in the class and feature subgraphs
N <- 10000      # Size of the simulated data set

# C1, ..., Cd
classes = sapply(1:d, function(x) paste("C", x, sep=""))
# X1, ..., Xm
features = sapply(1:(m+s), function(x) paste("X", x, sep=""))

# Number of experiments
executions <- 1
for (case in 1:executions) {
  print(paste0("Execution ", case))
  # Simulate MBCTree and data set
  MBCTree_init <- random_MBCTree(features, classes, s, parents)
  info_MBCTree(MBCTree_init)
  print("-------")
  dataset <- sample_MBCTree(MBCTree_init, N)

  # Split data set
  Nreal <- nrow(dataset) # It may not be equal to N because of how sample_MBCTree is implemented
  dataset <- dataset[sample(Nreal), ] # Shuffle
  training <- 0.6
  test <- 0.2
  validation <- 0.2
  training_set <- dataset[1:(training*Nreal), ]
  test_set <- dataset[(training*Nreal+1):((training+test)*Nreal), ]
  validation_set <- dataset[(((training+test)*Nreal)+1):Nreal, ]
  
  ## Performance of MBC
  # Learn network and parameters
  MBC <- learn_MBC(rbind(training_set, validation_set), classes, features)
  # Predict
  out <- predict_MBC_dataset_veryfast(MBC, test_set, classes, features)
  # Performance
  print("MBC")
  performance_MBC <- test_multidimensional(test_set, out, classes)
  print(performance_MBC$global)
  print("-------")
  
  ## Performance of MBCTree
  # Learn MBCTree
  MBCTree <- learn_MBCTree(training_set, validation_set, classes, features, verbose=TRUE)
  print("-------")
  info_MBCTree(MBCTree)
  print("-------")
  # Predict
  out <- predict_MBCTree_dataset(MBCTree, test_set, classes, features)
  # Performance
  print("MBCTree")
  performance_MBCTree <- test_multidimensional(test_set, out, classes)
  print(performance_MBCTree$global)
  print("========")
}
