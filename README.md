# LiDAR-PointClouds-Classification



## 3D Matching of TerraSAR-X Derived Ground Control Points to Mobile Mapping Data

*   Sponsored by: [Airbus Defence and Space GmbH, Friedrichshafen](/index.php?id=41&L=0#c692)
*   Project Lead: Dr. Ricardo Acevedo Cabra
*   Scientific Lead: [Dr. Wolfgang Koppe](mailto:wolfgang.koppe@airbus.com "Opens window for sending email") (Phone: 07545/8-4226) and [Tatjana Bürgmann](mailto:tatjana.buergmann@airbus.com "Opens window for sending email") (07545/8-1431)
*   Term: Summer semester 2019


 ![Classified in-situ Lidar data, overlayed by TerraSAR-X derived GCPs (red points)  
 Source: Airbus Defence and Space, ZF](/img/csm_Airbus_SS2019_Project_image_649e058fda.png)
 
 *Fig. 1: Classified in-situ Lidar data, overlayed by TerraSAR-X derived GCPs (red points)  
 Source: Airbus Defence and Space, ZF*


Radar imaging satellites like TerraSAR-X are able to acquire images having very high absolute geo-location accuracy, due the availability of precise orbit information. By using multiple stereo images in a radargrammetric process, so-called Ground Control Points (GCPs) can be extracted. GCPs are precisely measured land marks given the exact position on the earth. These GCPs are derived from pole-like structures along the road e.g. street lights, signs or traffic lights, since these objects are having a high backscatter in the radar image and therefore being easily identifiable in multiple images. By using a stack of multiple TerraSAR-X images, a dense point cloud of GCPs having an accuracy of less than 10 centimeters can be automatically extracted.

However, in order to make use of this high positional accuracy for the use case of autonomous driving, the link between landmarks like street lights identified from mobile mapping data and the coordinates of the respective GCP needs to be established. The goal of this project is to find and implement an algorithm for the automatic matching of 3D point clouds from GCPs extracted by radar space geodesy and in-situ LIDAR mobile mapping data derived from a car acquisition. A precise matching process would enable the generation of an accurate data basis as indispensable basis for highly automated and autonomous driving.

The particular tasks involve an initial literature study to gather possible approaches and solutions, performing a feasibility analysis of these approaches, the implementation of one or possibly two different approaches, and the evaluation of the developed method compared to a baseline method.







Papers:
1. [KIT Software and Datasets](http://www.ipf.kit.edu/code.php)
2. [SEMANTIC 3D SCENE INTERPRETATION: A FRAMEWORK COMBINING OPTIMAL
NEIGHBORHOOD SIZE SELECTION WITH RELEVANT FEATURES](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/II-3/181/2014/isprsannals-II-3-181-2014.pdf)
3. [Classifying Single Photon LiDAR data using Machine Learning and Open Source tools](http://www.eurosdr.net/sites/default/files/images/inline/10_garcia-morales.pdf)



Helpfull Repo:

1. [ecolidar_knowledgebase](https://github.com/eEcoLiDAR/ecolidar_knowledgebase)
2. [point-cloud-processing](https://github.com/rockestate/point-cloud-processing)
3. [Risk_Detection_UI](https://github.com/HaroldMurcia/Risk_Detection_UI/)
4. [Dissertation](https://github.com/NoemiRoecklinger/dissertation/blob/5e235617296910075af3444fc3edc3ea589c1843/4_CreateAllFeatures_subset1000.ipynb)
5. [Custom python filters for pdal](https://github.com/ArcticSnow/photo4D/blob/5c76c0256e54ad80c2f4cac96827e43f7ba214d8/build/lib/photo4d/pdal_python_filter.py)
6. [Class and functions to process the point clouds](https://github.com/ArcticSnow/photo4D/blob/master/photo4d/Class_pcl_processing.py)
7. [Features3D](https://github.com/HaroldMurcia/Risk_Detection_UI/blob/ad03ec6baf9a789f3fe889895cfb665134818ac9/Dev_Python/Risk_Detection/Features3D.py)
8. [Entwine, which allows you to organize very large collections of data as an octree, parallelize the process](https://entwine.io/configuration.html)


