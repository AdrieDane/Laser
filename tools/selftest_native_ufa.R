# Synthetic native-engine controls only; no study spectra or reference labels.
# Rscript selftest_native_ufa.R ISOLATED_R_LIBRARY
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
.libPaths(c(args[1],.libPaths()))
library(IDSL.UFA)
for (adduct in c("[M+H]+","[M-H]-","[M-2H]2-")) {
  db <- molecularFormula2IPdb(c("C42H82NO8P","C43H86NO7P"),IonPathways=adduct,
    peak_spacing=.005,intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  isotope <- db$IsotopicProfile[[1]]
  charge <- if (adduct=="[M+H]+") 1 else if (adduct=="[M-H]-") -1 else -2
  # Native atom-sum masses omit the electron correction. Feed physically
  # charged-ion masses to verify that the pinned native tolerance handles it.
  observed <- isotope[,1]-sign(charge)*.000548579909/abs(charge)
  shape <- c(.01,.03,.1,.3,.6,1,.6,.3,.1,.03,.01)
  spectra <- lapply(shape,function(a)cbind(observed,a*isotope[,2]*1000))
  pl <- matrix(0,nrow=1,ncol=11)
  pl[1,c(1,2,3,4,8,11)] <- c(1,length(shape),.5,100000,
      db$MassMAIso[1]-sign(charge)*.000548579909/abs(charge),db$R13C[1])
  ans <- molecular_formula_annotator(db,spectra,pl,1,massAccuracy=.01,maxNEME=5,
      minPCS=950,minNDCS=2,minRCS=5,scoreCoefficients=rep(1,5),number_processing_threads=1)
  stopifnot(nrow(ans)>0,ans$IonFormula[1]==db$MolecularFormula[1])
  cat(adduct,"synthetic top-1 passed\n")
}
cat("NATIVE UFA SYNTHETIC CONTROLS PASSED\n")
