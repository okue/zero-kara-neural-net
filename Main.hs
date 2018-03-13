{-# LANGUAGE BangPatterns, MultiWayIf, LambdaCase #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts  #-}
module Main where

import Numeric.LinearAlgebra
import System.Random (randomRIO)

import qualified NeuralNet.Mnist as Mnist
import NeuralNet.Types
import NeuralNet.Utils
import NeuralNet.Layers

main :: IO ()
main = do
  let
    weight_init_std = 0.1
    learning_rate = 0.01
    batch_size  = 100
    input_size  = 784
    hidden_size = 50
    output_size = 10
    train_num   = 10000
  [(xs, ys), (verify_xs, verify_ys)] <- Mnist.loadMnist True
  !afl1 <- mkAffineLayer weight_init_std input_size hidden_size
  !afl2 <- mkAffineLayer weight_init_std hidden_size output_size
  !r <- genRandList 10000 1000
  let
    !layers = [ afl1, ReLULayer Nothing, afl2 ]
    !nn = NeuralNet { _layers = layers, _lastlayer = (`SoftmaxLayer` Nothing) }
    !vx = verify_xs ? r
    !vy = verify_ys ? r
    !train_size = length $ toRows xs
  !trained_nn <- train nn xs ys train_size batch_size learning_rate 0 train_num
  let predict_list = predict trained_nn vx
  putStr "percentage of correct answers "
  print . (/ 10) . fromIntegral . length . filter (True ==) $ zipWith (\a b -> maxIndex a == ceiling b) (toRows predict_list) (aux vy)
    where
      aux = toList . flatten
