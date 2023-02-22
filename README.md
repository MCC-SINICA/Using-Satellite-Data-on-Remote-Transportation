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

### EPA Data
For 2014 to 2018, all of the stations with weather/polluton in Taiwan can download on the following cloud drives.
https://1drv.ms/u/s!AmHd3ERrMbP0wbg1TLI-FfEUN-Qbiw?e=8FsXYI

### EPA Preprocessing Data
For 2014 to 2018, 18 stations in Taipei Area. Each of which contain 14 features.
0: AMB_TEMP , 1: CO, 2: NO, 3: NO2, 4: NOx, 5: O3, 6: PM10, 7: PM2.5, 8: Rainfall, 9: RH, 10: SO2
11: THC, 12: COS_wind, 13: Sin_wind.

The numpy shape of preprocessing data is (8760,14,18), where 8760 indicate 365(days)x24(hr), 14 indicate features
18 indicate stations.

To get weather data, use following python command:
weather_array_14 = np.concatenate(np.load('epa14_18station.npy')[:,1:8,:], np.load('epa14_18station.npy')[:,10:12,:], axis=1)

To get pollution data, use following python command:
pollution_array_14 = np.concatenate(np.load('epa14_18station.npy')[:,0:1,:], np.load('epa14_18station.npy')[:,8:10,:], axis=1)

Preprocessing data can be found in following path.

https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation/blob/main/data/epa14_18station.npy
https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation/blob/main/data/epa15_18station.npy
https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation/blob/main/data/epa16_18station.npy
https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation/blob/main/data/epa17_18station.npy
https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation/blob/main/data/epa18_18station.npy


## INSTALL


- python 3.6
- keras
- numpy
- scipy
- matplotlib
