e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_classified_part2.e57" -o "D:\repo\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_.las" -split_scans

e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_classified_part2.e57" -o "D:\repo\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_.las" -split_scans

e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_classified_part2.e57" -o "D:\repo\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_.las" -split_scans

e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_part1.e57" -o "D:\repo\3D-Matching-Prj\data\export\classes\Werk2_part1.las" 

e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_part2.e57" -o "D:\repo\3D-Matching-Prj\data\export\classes\Werk2_part2_.las" 

e572las -v -i "D:\repo\3D-Matching-Prj\data\export\Werk2_part3.e57" -o "D:\repo\3D-Matching-Prj\data\export\raw\Werk2_part3_.las" 


las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_classified_part1.las" -o "D:\repo\new\3D-Matching-Prj\data\export\*.las" -epsg 32632

las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part1.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part1.shp"
 
las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part2.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part2.shp"

las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part3.las" -o  "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part3.shp"


las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part1.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part1.las" -epsg 32632
 
las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part2.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part2.las" -epsg 32632

las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_part3.las" -o  "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part3.las" -epsg 32632

las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_classified_part1.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_classified_part1.las" -epsg 32632
las2las -i "D:\repo\new\3D-Matching-Prj\data\tracks\Werk2_classified_part2.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_classified_part2.las" -epsg 32632


las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part1.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part1.shp"
las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part2.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part2.shp"
las2shp -i "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part3.las" -o "D:\repo\new\3D-Matching-Prj\data\tracks\projected\Werk2_part3.shp"

lasmerge -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00000.las" -set_classification 0 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00001.las" -set_classification 1 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00002.las" -set_classification 2 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00003.las" -set_classification 3 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00004.las" -set_classification 4 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00005.las" -set_classification 5 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00006.las" -set_classification 6 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00007.las" -set_classification 7 "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00008.las" -set_classification 8 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1.las"


las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00000.las" -change_classification_from_to 0 7  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_0.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00001.las" -change_classification_from_to 0 2  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_1.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00002.las" -change_classification_from_to 0 11 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_2.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00003.las" -change_classification_from_to 0 3  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_3.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00004.las" -change_classification_from_to 0 6  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_4.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00005.las" -change_classification_from_to 0 19 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_5.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00006.las" -change_classification_from_to 0 20 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_6.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00007.las" -change_classification_from_to 0 21 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_7.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00008.las" -change_classification_from_to 0 22 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part1_9.las"


lasmerge -i "D:\repo\new\3D-Matching-Prj\data\*.las" -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1.las"

lasmerge -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00000.las" -set_classification 0 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00001.las" -set_classification 1 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1_.las"

lasmerge -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00000.las" -change_classification_from_to 0 0 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00001.las" -change_classification_from_to 0 1 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1_.las"


lasmerge -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00000.las" -set_classification 0 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00001.las" -set_classification 1 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00002.las" -set_classification 2 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00003.las" -set_classification 3 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00004.las" -set_classification 4 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00005.las" -set_classification 5 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00006.las" -set_classification 6 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00007.las" -set_classification 7 -epsg 32632 -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part1_00008.las" -set_classification 8 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1.las"


# Part two:
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00000.las" -change_classification_from_to 0 7  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2_00.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00001.las" -change_classification_from_to 0 7  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\\Werk2_classified_part2_000.las"
lasmerge -i "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part2_00.las" "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part1\Werk2_classified_part2_000.las" -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_0.las"

las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00002.las" -change_classification_from_to 0 2  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_1.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00003.las" -change_classification_from_to 0 11 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_2.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00004.las" -change_classification_from_to 0 3  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_3.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00005.las" -change_classification_from_to 0 6  -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_4.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00006.las" -change_classification_from_to 0 19 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_5.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00007.las" -change_classification_from_to 0 20 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_6.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00008.las" -change_classification_from_to 0 21 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_7.las"
las2las -i "D:\repo\new\3D-Matching-Prj\data\export\classes\Werk2_classified_part2_00009.las" -change_classification_from_to 0 22 -epsg 32632 -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\Werk2_classified_part2_9.las"


lasmerge -i "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2\*.las" -o "D:\repo\new\3D-Matching-Prj\data\Werk2_classified_part2.las"