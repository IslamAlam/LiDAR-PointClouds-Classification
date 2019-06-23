from pathlib import Path
import time
import pandas as pd


import json
import pdal


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--inputfile", dest="inputfilename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-o", "--outputfile", dest="outputfilename",
                    help="write report to FILE", metavar="FILE")

parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")

args = parser.parse_args()
print(args)
print(args.inputfilename)

inputfile = str(args.inputfilename)

#outputfile = inputfile[:-4] + "_features" + inputfile[-4:]
outputfile = str(args.outputfilename)


pipe_reader =\
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
        "type":"writers.bpf",
        "filename":outputfile,
        "output_dims":"X,Y,Z,HeightAboveGround,Classification,NNDistance,Eigenvalue0,Eigenvalue1,Eigenvalue2,NormalX, NormalY,NormalZ,Curvature,Rank,RadialDensity,Coplanar"
    }
  ]
}


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
#pipeline = pdal.Pipeline(json.dumps(pipe_features))
pipeline = pdal.Pipeline(json.dumps(pipe_add_features))

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

"""
   {
        "type":"writers.las",
        "vlrs": [{
            "description": "A description under 32 bytes",
            "record_id": 42,
            "user_id": "hobu",
            "data": "dGhpcyBpcyBzb21lIHRleHQ="
            },
            {
            "description": "A description under 32 bytes",
            "record_id": 43,
            "user_id": "hobu",
            "data": "dGhpcyBpcyBzb21lIG1vcmUgdGV4dA=="
            }],
        "filename":outputfile
    }
    """
    