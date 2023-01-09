# longterm-transported-pollutants

The name of paper is **Using Satellite Data on Remote Transportation of Air Pollutants for PM<sub>2.5</sub> Prediction in Northern Taiwan**

In this repository we provide Code files in Python language to reproduce results of the mention paper. for capturing longterm transported pollutant(PM<sub>2.5</sub>) and used them to improve local prediction of PM<sub>2.5</sub>

The project involve the prediction of PM<sub>2.5</sub> for next 4hour(hr), 8hr,12hr,.. to 72hr. These codes performed prediction 
for next 4hour. The same codes are used from predicting the remaining hours but same changes needed to the 
data shape of input features and output label data.

STRI_fe code with extract middle layer used to extract remote pollutants

STR_p code use the extracted remote pollutants with other features as input for local prediction of PM<sub>2.5</sub>.

RTP composite code performed the final prediction of PM<sub>2.5</sub> by considering prediction results from STRI_p model and the base model

Most of these code are executed in the terminal e.g `# $ python3 next4hr_stri_fe_model_.py > out_stri_fe_n4hr.log`

## DATA

Majority of the dataset we used are very large for example train size for each satellite tile is 11.7GB, remote weather train size is 1.1GB and for testing each tile has a size of 5.9GB while remote weather is 555MB. Therefore we were not able to include those file in our repository.

## INSTALL


- python 3.6
- keras
- numpy
- scipy
- matplotlib

