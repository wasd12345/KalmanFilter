# KalmanFilter
Kalman Filter (linear, not extended or unscented)



Defines the class KalmanFilter and tests it on simple demo.

The demo is a tracking problem on the xy plane, with random kinematics
up to constant jerk (3rd derivative of position).

The goal is to track the object's kinematics when you only get noisy 
measurements of x,y position and x,y acceleration (maybe the velocity sensor is broken).
Also, maybe you don't have sensors to measure jerk, or maybe they are nonfunctional.



Assumptions of this Kalman Filter:
- State Transition model is linear function of previous state
  (-> can be modeled as matrix multiply of State Transition Matrix on 
  previous state vector)
- Process noise and Measurement noise are both Gaussian, and uncorrelated

If the transition model is NONlinear, can possibly linearize it using
Extended Kalmana Filter.
If highly NONlinear, possibly can work with Unscented Kalman Filter.
In either case, possibly best to just skip and try particle filter. 
With ~infinity Nparticles -> optimal Bayesian MSE estimator
