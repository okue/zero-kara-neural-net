module NeuralNet.Types where

import Numeric.LinearAlgebra

type M  = Matrix R
type MM = Maybe M
type Layers = [Layer]
data NeuralNet = NeuralNet { _layers :: Layers, _lastlayer :: M -> Layer }
data Layer = AffineLayer { _w :: M, _b :: M, _x :: MM, _dw :: MM, _db :: MM }
           | ReLULayer { _x :: MM }
           | SigmoidLayer MM
           | SoftmaxLayer { _t :: M, _y :: MM }
           deriving Show
