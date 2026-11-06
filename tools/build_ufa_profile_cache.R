# Build one native profile universe per adduct, shared by every acquisition mode.
# Rscript build_ufa_profile_cache.R CONFIG.json NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
config <- jsonlite::read_json(args[1],simplifyVector=FALSE)
.libPaths(c(config$r_library,.libPaths()))
library(IDSL.UFA)
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),"ufa_trace_lib.R"))
output <- args[2];if (dir.exists(output)) stop("Preserve previous cache builds")
dir.create(output,recursive=TRUE)
by_adduct <- list()
for (path in unlist(config$mode_configs)) {
  mode <- jsonlite::read_json(path,simplifyVector=FALSE)
  for (q in mode$queries) {
    # Cache ALL generated candidates, including queries with unavailable raw
    # peaks. Candidate generation/coverage must not depend on successful traces.
    pool <- vapply(q$candidates,function(c)c$formula,character(1))
    by_adduct[[q$adduct]] <- unique(c(by_adduct[[q$adduct]],pool))
  }
}
began <- proc.time()[[3]];manifest <- list(schema=1L,spec=ufa_cache_spec(),adducts=list())
for (adduct in sort(names(by_adduct))) {
  if (!is.null(config$adduct_subset) && !adduct%in%unlist(config$adduct_subset)) next
  cat("Native profile cache:",adduct,length(by_adduct[[adduct]]),"unique formulas\n");flush.console()
  cache <- ufa_cached_database(by_adduct[[adduct]],adduct,file.path(output,"profiles"))
  probes <- sort(unique(by_adduct[[adduct]]))
  probes <- probes[unique(round(seq(1,length(probes),length.out=min(8L,length(probes)))))]
  reference <- IDSL.UFA::molecularFormula2IPdb(probes,IonPathways=adduct,peak_spacing=.005,
      intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  stopifnot(identical(reference,ufa_subset_database(cache$db,reference$MolecularFormula)))
  manifest$adducts[[adduct]] <- cache[setdiff(names(cache),"db")]
  manifest$adducts[[adduct]]$requested_formulas <- length(by_adduct[[adduct]])
  manifest$adducts[[adduct]]$native_profiles <- length(cache$db$MolecularFormula)
  manifest$adducts[[adduct]]$unmodified_native_probe_parity <- length(probes)
}
manifest$seconds <- proc.time()[[3]]-began
jsonlite::write_json(manifest,file.path(output,"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
cat("ALL NATIVE PROFILE CACHES COMPLETE\n")
