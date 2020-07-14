1) create a way for the structure names and dimensions to propagate all the way throgh the pipeline
2) fix the normalisation of DOS by bin width
3) rework job scripts to save their raw data in workspace and only the processed data into ~/data
4) rework the scripts so that infinite temperature models are loaded in using the same IPR and DOS code as MCMC data, or alternatively, factor out that code into a separate function for both.