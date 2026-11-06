# Synthetic format, optimization-equivalence and runtime gate; no study data.
# Rscript pilot_ufa_bridge.R ISOLATED_R_LIBRARY NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)%in%c(2L,3L))
.libPaths(c(args[1],.libPaths()));library(IDSL.UFA)
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),"ufa_trace_lib.R"))
initial_spec <- ufa_cache_spec()
out <- args[2];if (dir.exists(out)) stop("Preserve earlier pilots")
dir.create(out,recursive=TRUE)
cases <- list(c("C42H82NO8P","[M+H]+","1"),c("C42H82NO8P","[M-H]-","-1"),
  c("C73H142O17P2","[M-2H]2-","-2"),c("C120H236O6","[M+NH4]+","1"),
  c("C120H236O6","[M-H]-","-1"),c("C120H236O6","[M-2H]2-","-2"),
  c("C48H93NO11S","[M-H]-","-1"),c("C42H82NO8P","[M+Cl]-","-1"))
shape <- c(.01,.03,.1,.3,.6,1,.6,.3,.1,.03,.01)
controls <- list();last <- NULL
for (case in cases) {
  formula <- case[1];adduct <- case[2];charge <- as.numeric(case[3])
  cold <- ufa_cached_database(formula,adduct,file.path(out,"profiles"))
  warm <- ufa_cached_database(formula,adduct,file.path(out,"profiles"))
  stopifnot(!cold$cache_hit,warm$cache_hit,identical(cold$db,warm$db),cold$sha256==warm$sha256)
  db <- ufa_subset_database(warm$db,warm$db$MolecularFormula)
  stopifnot(identical(db,warm$db))
  reference <- IDSL.UFA::molecularFormula2IPdb(formula,IonPathways=adduct,peak_spacing=.005,
      intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  stopifnot(identical(reference,db))
  ip <- db$IsotopicProfile[[1]]
  observed <- ip[,1]-sign(charge)*.000548579909/abs(charge)
  spectra <- lapply(shape,function(a) {
    # Off-target centroids dominate scan size but must not change a native hit.
    noise <- cbind(seq(150,2000,length.out=1500),rep(1,1500))
    rbind(cbind(observed,a*ip[,2]*1000),noise)
  })
  anchor <- ufa_observed_anchor(spectra,observed[1],charge,times=seq_along(shape))
  expected <- round((db$MassMAIso[1]-ip[1,1])*abs(charge)/1.003354835336)
  stopifnot(anchor$order==expected)
  unpruned <- ufa_annotate(db,spectra,anchor,.5,prune=FALSE)
  pruned <- ufa_annotate(db,spectra,anchor,.5,prune=TRUE)
  stopifnot(!is.null(pruned),pruned$IonFormula[1]==db$MolecularFormula[1],identical(unpruned,pruned))
  controls[[length(controls)+1L]] <- list(formula=formula,adduct=adduct,charge=charge,
      observed_anchor_order=anchor$order,native_top1_correct=TRUE,cold_warm_exact=TRUE,
      pruned_unpruned_exact=TRUE)
  last <- list(db=db,spectra=spectra,mono=observed[1],charge=charge,anchor=anchor)
}
expect_unavailable <- function(spectra,why) {
  value <- tryCatch(ufa_observed_anchor(spectra,last$mono,last$charge),error=function(e)e)
  if (!inherits(value,"error")) stop(paste("Interference control did not abstain:",why))
  cat(why,conditionMessage(value),"\n")
}
expect_unavailable(last$spectra[c(1,6,11)],"sparse scans")
duplicate <- lapply(last$spectra,function(s)rbind(s,c(last$mono+.0001,1e5)))
expect_unavailable(duplicate,"competing monoisotopic centroids")
interference <- lapply(seq_along(shape),function(i) {
  s <- last$spectra[[i]]
  target <- last$mono+1.003354835336/abs(last$charge)
  s <- s[abs(s[,1]-target)>target*4e-6,,drop=FALSE]
  rbind(s,c(target,2e5*shape[max(1,i-3)]))
})
expect_unavailable(interference,"shifted strong isotope interference")

# Nearest-centroid ties must survive the exact lossless filtering operation.
db <- last$db;mass <- db$IsotopicProfile[[1]][1,1]
ties <- lapply(last$spectra,function(s)rbind(c(mass-.001,1),c(mass+.001,2),s))
filtered <- ufa_prune_spectra(ties,db)
stopifnot(all(vapply(filtered,function(s)identical(unname(s[1:2,]),unname(matrix(c(mass-.001,mass+.001,1,2),ncol=2))),logical(1))))
stopifnot(identical(ufa_annotate(db,ties,last$anchor,.5,FALSE),ufa_annotate(db,ties,last$anchor,.5,TRUE)))

# Fixed 512-profile chemical-space timing probe, including high C and sulfur.
grid <- expand.grid(C=seq(24,120,4),O=c(2,4,6,8,10,12,16),S=c(0,1,8))
grid <- grid[unique(round(seq(1,nrow(grid),length.out=512))),]
pool <- sprintf("C%dH%dO%d%s",grid$C,2*grid$C-2,grid$O,ifelse(grid$S>0,paste0("S",grid$S),""))
began <- proc.time()[[3]]
cold <- ufa_cached_database(pool,"[M+H]+",file.path(out,"timing"))
cold_seconds <- proc.time()[[3]]-began
if (length(args)==3L) {
  reference <- readRDS(args[3])
  stopifnot(identical(reference$formulas,sort(unique(pool))),identical(reference$db,cold$db))
}
for (formula in c("C60H110N4O24P2S3","C20H41N6O60P3S3")) {
  reference <- IDSL.UFA::molecularFormula2IPdb(formula,IonPathways="[M+H]+",peak_spacing=.005,
      intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  optimized <- UFA_EXACT_GENERATOR(formula,IonPathways="[M+H]+",peak_spacing=.005,
      intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  stopifnot(identical(reference,optimized))
}
began <- proc.time()[[3]]
warm <- ufa_cached_database(pool,"[M+H]+",file.path(out,"timing"))
warm_seconds <- proc.time()[[3]]-began
stopifnot(identical(cold$db,warm$db),warm$cache_hit)
began <- proc.time()[[3]]
for (i in 1:100) stopifnot(!is.null(ufa_annotate(last$db,last$spectra,last$anchor,.5)))
scoring_100_seconds <- proc.time()[[3]]-began
passed <- cold_seconds<=30 && warm_seconds<=5 && scoring_100_seconds<=10
stopifnot(identical(initial_spec,ufa_cache_spec()))
receipt <- list(controls=controls,interference_abstention_controls=3L,centroid_tie_order_exact=TRUE,
  native_profiles_512_cold_seconds=cold_seconds,native_profiles_512_warm_seconds=warm_seconds,
  scoring_100_seconds=scoring_100_seconds,correctness_passed=TRUE,runtime_gate_passed=passed,
  gate=list(cold_512_max_seconds=30,warm_512_max_seconds=5,scoring_100_max_seconds=10),
  anchor_settings=UFA_ANCHOR_SETTINGS,cache_spec=ufa_cache_spec(),study_data_used=FALSE,
  native_unmodified_profile_parity=TRUE,cache_descriptor_stable=TRUE,
  profile_parity_N=if(length(args)==3L)522L else 10L)
jsonlite::write_json(receipt,file.path(out,"receipt.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
if (!passed) stop("Synthetic runtime gate failed; do not launch full study scoring")
cat("ALL SYNTHETIC CORRECTNESS, CACHE, LOSSLESS-PRUNING AND RUNTIME GATES PASSED\n")
