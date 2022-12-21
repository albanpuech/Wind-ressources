# Wind energy resources vizualisation platform

We present a dashboard-like visualization interface that aims at making the spatial and temporal variability of wind power more accessible to decision-makers. We also hope to bridge the gap between the available climate data and the energy industry looking for more intelligible analysis tools.
## Link
http://albanpuech.eu.pythonanywhere.com/
## Dataset 
The dataset used here is an ERA5-derived time series of European country-aggregate electricity wind power generation [1]. The authors use the ERA5 reanalysis data [2] to compute the nationally aggregated hourly capacity factor of 28 European countries. The capacity factor of a wind turbine is the ratio of the electrical energy produced by the turbine for the period of time considered to the total amount of electrical energy that could have been produced at full power operation during the same period. Reanalysis combines past weather observations and current atmospheric models to generate climate and weather historic data. It allows getting a complete weather record from sparse - both in space and time - past data. In addition to the wind speed data, the authors use the wind farm spatial distribution of 2017, taken from https://thewindpower.net.
However, it is worth mentioning that, because the absolute wind power capacity is not used to compute the capacity factors, only the relative spatial distribution of wind turbines is assumed to be constant. The capacity factor of each country is estimated by aggregating the capacity factor computed for each grid box, weighted by its estimated installed capacity. The capacity factor in each grid box is derived using the 100m wind speeds and the power curve of the type of wind turbine maximizing the energy produced during the entire period (1979-2019), as indicated in [3].
## References
[1] H. Bloomfield, D. Brayshaw, and A. Charlton-Perez, “Era5 derived time series of european country-aggregate electricity demand, wind power generation and solar power generation: hourly data from 1979-2019,” 2020

[2] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara,A. Hor ́anyi, J. Mu ̃noz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, A. Simmons, C. Soci, S. Abdalla, X. Abellan, G. Balsamo, P. Bechtold, G. Biavati, J. Bidlot, M. Bonavita, G. De Chiara, P. Dahlgren, D. Dee, M. Diamantakis, R. Dragani, J. Flemming, R. Forbes, M.Fuentes, A. Geer, L. Haimberger, S. Healy, R. J. Hogan, E. H ́olm, M. Janiskov ́a, S. Keeley, P. Laloyaux, P. Lopez, C. Lupu, G. Radnoti, P. de Rosnay, I. Rozum, F. Vamborg, S. Villaume, and J.-N. Th ́epaut, “The era5 global reanalysis,” Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730,pp. 1999–2049, 2020

[3] H. Bloomfield, D. Brayshaw, D. Brayshaw, P. Gonzalez, P. Gonzalez, and A. Charlton-Perez, “Sub-seasonal forecasts of demand and wind power and solar power generation for 28 european countries”, Earth System Science Data, vol. 13, pp. 2259–2274,05 2021

