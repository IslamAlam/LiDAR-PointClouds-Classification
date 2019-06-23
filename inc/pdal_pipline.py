from pathlib import Path
import time
import pandas as pd


import json
import pdal


def las_2_df(las_path):
    pipe_LASreader =\
    {
      "pipeline":[
        {
          "type":"readers.las",
          "filename":las_file,
          "use_eb_vlr": "true"
        }
      ]
    }
    # """%lasinput
    #print(pipe_reader)

    pipeline = pdal.Pipeline(json.dumps(pipe_LASreader))
    pipeline.validate()
    print(pipeline.validate())
    # %time n_points = pipeline.execute()
    start_time = time.time()
    n_points = pipeline.execute()
    elapsed_time_fl = (time.time() - start_time)
    lidar_df = pd.DataFrame(pipeline.arrays[0])
    print("Number of points in LiDAR:", n_points)
    #print(lidar_df)
    lidar_df.head()
    
    return lidar_df

def las_add_features(las_path):
    inputfile = str(las_path)
    pipe_features =\
    {
      "pipeline":[
        {
          "type":"readers.las",
          "filename":inputfile
        },
          {
            "type":"filters.lof",
            "minpts":20
          },
          {
            "type":"filters.nndistance",
            "k":8
          },
          {
            "type":"filters.eigenvalues",
            "knn":8
          },
          {
            "type":"filters.estimaterank",
            "knn":8,
            "thresh":0.01
          },
          {
            "type":"filters.normal",
            "knn":8
          },
          {
            "type":"filters.radialdensity",
            "radius":2.0
          },
          {
            "type":"filters.approximatecoplanar",
            "knn":8,
            "thresh1":25,
            "thresh2":6
          },
          {
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
          },
          {
            "type":"filters.hag"
          },
          {
          "type":"filters.elm",
          "cell":20.0,
          "class":7,
          "threshold":1.0
          },
          {
            "type":"writers.las",
            "filename":outputfile,
            "minor_version": "4",
            "extra_dims":"all"
        }
      ]
    }

    pipe_add_features =\
    {
      "pipeline":[
        {
          "type":"readers.las",
          "filename":inputfile
        },
          {
            "type":"filters.lof",
            "minpts":20
          },
          {
            "type":"filters.nndistance",
            "k":8
          },
          {
            "type":"filters.eigenvalues",
            "knn":8
          },
          {
            "type":"filters.estimaterank",
            "knn":8,
            "thresh":0.01
          },
          {
            "type":"filters.normal",
            "knn":8
          },
          {
            "type":"filters.radialdensity",
            "radius":2.0
          },
          {
            "type":"filters.approximatecoplanar",
            "knn":8,
            "thresh1":25,
            "thresh2":6
          },
          {
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
          },
          {
            "type":"filters.hag"
          },
          {
          "type":"filters.elm",
          "cell":20.0,
          "class":7,
          "threshold":1.0
          },
          {
              "type":"filters.python",
              "script":"PCFeatures3D.py",
              "function":"calcFeatureDescr",
              "add_dimension":['Linearity', 'Planarity', 'Scattering','Omnivariance', 'Anisotropy',
    'Eigentropy', 'Eigen_Sum', 'Curvature_Change'],
              "module": "anything"

          },
          {
            "type":"writers.las",
            "filename":outputfile,
            "extra_dims":"all"
        }
      ]
    }

    #pipeline = pdal.Pipeline(json.dumps(pipe_reader))
    pipeline = pdal.Pipeline(json.dumps(pipe_features))
    #pipeline = pdal.Pipeline(json.dumps(pipe_add_features))

    pipeline.validate()
    print(pipeline.validate())

    #%time 
    start_time = time.time()
    n_points = pipeline.execute()
    elapsed_time_fl = (time.time() - start_time)
    print('Time taken ', elapsed_time_fl,' seconds')

    print("Number of points in LiDAR:", n_points)

    lidar_df = pd.DataFrame(pipeline.arrays[0])
    print(lidar_df)
    lidar_df.head()