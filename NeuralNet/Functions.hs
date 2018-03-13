{-# LANGUAGE BangPatterns, MultiWayIf, LambdaCase #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts  #-}
module NeuralNet.Functions (
  step,
  sigmoid,
  relu,
  softmax,
  meanSquaredError,
  crossEntropyError,
) where

import Numeric.LinearAlgebra

import NeuralNet.Types


sigmoid :: R -> R
sigmoid a = 1 / (1 + exp (-a))

relu :: R -> R
relu a = a `max` 0

softmax :: M -> M
softmax x = fromRows . map aux . toRows $ x
  where
    aux :: Vector R -> Vector R
    aux a =
      let
        !c = maxElement a
        !exp_a = cmap (\x -> exp (x - c)) a
        !sum_exp_a = sumElements exp_a
      in exp_a / scalar sum_exp_a

meanSquaredError :: M -> M -> R
meanSquaredError y t = (0.5 *) . sumElements $ cmap (**2) (y - t)

crossEntropyError :: M -> M -> R
crossEntropyError y t = sumElements $ y * cmap (\a -> - log (a + 1e-7)) t
