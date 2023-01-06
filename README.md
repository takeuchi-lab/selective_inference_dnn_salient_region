# Instrattion & Requirements
We used Python 3.10.2.  To run this package, please install dependencies with following code.

```
pip install requirements.txt
pip install -e .
```

# Reproducibility
Since we have already got the results in advance, you can reproduce the figures by running following code. 
The results will be save in "/plot" folder.
```
cd plot
python plot.py
```

To reproduce the results, please see the following instructions　after a Instrattion step.
For the setting of one reference image, please run
All the results will be save the "./experiment_result" folder as csv file.

For the setting of one reference images, please run
```
./ex1_one_reference_image.sh
```

For the setting of multiple reference images, please run
```
./ex2_multiple_reference_image.sh
```