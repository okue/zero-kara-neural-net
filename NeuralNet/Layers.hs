{-# LANGUAGE BangPatterns, MultiWayIf, LambdaCase #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts  #-}
module NeuralNet.Layers where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import Debug.Trace (trace)

import qualified NeuralNet.Functions as F
import qualified NeuralNet.Utils as U
import NeuralNet.Types


instance Show NeuralNet where
  show (NeuralNet l lastl) = "Layers:\n" ++ show l

class Forward a where forward :: a -> Matrix R -> (Matrix R, a)
class Backward a where backward :: a -> Matrix R -> (Matrix R, a)

instance Forward Layer where
  forward !l !x =
    case l of
      AffineLayer w b _ _ _ -> (x <> w + b, l { _x = Just x })
      SoftmaxLayer t _ ->
        let !y = F.softmax x
        in (scalar (F.crossEntropyError y t), l { _y = Just y })
      ReLULayer _    -> (cmap F.relu x, l { _x = Just x })
      SigmoidLayer _ -> (cmap F.sigmoid x, l)

instance Forward Layers where
  forward ls x = _forward ls x []
    where
      _forward [] x updated_layers = (x, updated_layers)
      _forward (l:ls) x updated_layers =
        let !(x', l') = forward l x
        in _forward ls x' (l':updated_layers)

instance Backward Layer where
  backward !l dout =
    case l of
      AffineLayer w b (Just x) _ _ ->
        let
          !dx = dout <> tr w
          !dw = tr x <> dout
          !db = fromColumns . map (scalar . sumElements) . toColumns $ dout
        in
          (dx, l { _dw = Just dw, _db = Just db })
      SoftmaxLayer t (Just y) ->
        let
          !batch_size = fromIntegral . fst $ size t
          !t' = fromRows . map (\x -> let i = ceiling (x!0) in U.mkVector i) $ toRows t
          !dx = (y - t') / batch_size
        in
          (dx, l)
      ReLULayer (Just x) ->
        let
          !dx = fromRows $ zipWith (zipVectorWith (\a b -> if b <= 0 then 0 else a)) (toRows dout) (toRows x)
        in
          (dx, l)
      SigmoidLayer (Just out) ->
        (dout * (1 - out) * out, l)

instance Backward Layers where
  backward ls dout = _backward ls dout []
    where
      _backward [] x updated_layers = (x, updated_layers)
      _backward (l:ls) x updated_layers =
        let !(x', l') = backward l x
        in _backward ls x' (l':updated_layers)

predict :: NeuralNet -> M -> M
predict nn x = fst $ forward (_layers nn) x

loss :: NeuralNet -> M -> M -> M
loss nn x t =
  let
    !(x', _) = forward (_layers nn) x
    !lastlayer = _lastlayer nn t
  in fst $ forward lastlayer x'

gradientLayers :: NeuralNet -> M -> M -> (R, Layers)
gradientLayers nn x t =
  let
    !(x', updated_layers)      = forward (_layers nn) x
    !(loss, updated_lastlayer) = forward (_lastlayer nn t) x'
    !(_, updated_layers')     = backward (updated_lastlayer:updated_layers) 1
  in (loss!0!0, init updated_layers')

train :: NeuralNet -> M -> M -> Int -> Int -> M -> Int -> Int -> IO NeuralNet
train nn xs ys train_size batch_size learning_rate i n
  | i == n = return nn
  | otherwise = do
      !r <- U.genRandList train_size batch_size
      let
        !x = xs ? r
        !t = ys ? r
        !(loss, gl) = gradientLayers nn x t
        !trained_nn = NeuralNet (map aux gl) (_lastlayer nn)
      if
        | i `mod` 300 == 0 -> putStrLn ("loss is " ++ show loss)
        | otherwise        -> return ()
      train trained_nn xs ys train_size batch_size learning_rate (i+1) n
  where
    aux af@(AffineLayer w b x (Just dw) (Just db)) =
      let _diff = sumElements (w - dw) in
      AffineLayer (w - learning_rate * dw) (b - learning_rate * db) Nothing Nothing Nothing
    aux x = x
