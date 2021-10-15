# slam-playground
Educational 2D SLAM implementation based on ICP and Pose Graph

![slamgui](https://github.com/Kolkir/slam-playground/blob/main/assets/slam.gif)

### How to use:

Use keyboard arrow keys to navigate robot. 

Press 'r' to view the raw map built from the odometry and sensor measurements.

Press 'i' to view map built from sensor measurements aligned with the ICP algorithm.

To update map and the robot path with the Pose Graph optimization, you have to navigate robot around the map and return it to the starting position in the center, the robot have to look in same direction as at the beginning. Then press the 's' or 'g' key.
The 's' key will start naive basic SLAM implementation and the 'g' key will launch solution based on [GTSAM](https://gtsam.org/) library.

To change the world map you can edit the 'map.png' file in the `assets` folder.

### Used resources

Please take a look in to the `doc` folder to find the reading list.

Also, you can find there the [jupyter notebook](https://github.com/Kolkir/slam-playground/blob/main/doc/slam_se2_derivation.ipynb) with derivation of all formulas for Jacobians and Hessian for SLAM implementation with corresponding Python examples.    

### How to configure:

You can set a different noise level for the odometry and sensors, or even disable noise at all. 
This configuration can be done by editing the `simulation.py` file, where you can change corresponding values for `mu` and `sigma` of a random gaussian noise.

```python
odometry = Odometry(mu=0, sigma=3)  # noised measurements
sensor = Sensor(dist_range=350, fov=90, mu=0, sigma=1)  # noised measurements
...
slam_back_end = playground.slam.backend.BackEnd(edge_sigma=0.5, angle_sigma=0.1)
```
