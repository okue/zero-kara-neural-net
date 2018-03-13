{-# LANGUAGE BangPatterns, MultiWayIf, LambdaCase #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts  #-}
module NeuralNet.Utils (
  mkAffineLayer,
  mkVector,
  genRandList,
) where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel

import NeuralNet.Types


mkAffineLayer :: M -> Int -> Int -> IO Layer
mkAffineLayer weight_init_std i_size o_size = do
  !m <- rand i_size o_size
  let
    !b = matrix o_size $ replicate o_size 0
    !w = weight_init_std * m
  return $ AffineLayer w b Nothing Nothing Nothing

mkVector :: Int -> Vector R
mkVector i = vector (replicate i 0 ++ [1] ++ replicate (9-i) 0)

genRandList :: Int -> Int -> IO [Int]
genRandList _mod batch_size = do
  !r <- rand 1 batch_size
  return . map (\x -> ceiling (1000000*x) `mod` _mod) . toList . flatten $ r
