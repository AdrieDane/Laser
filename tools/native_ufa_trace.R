# Native conditional IDSL.UFA reranking; shared candidate generation is external.
# Rscript native_ufa_trace.R PRIVATE_CONFIG.json NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
config <- jsonlite::read_json(args[1],simplifyVector=FALSE)
.libPaths(c(config$r_library,.libPaths()))
library(IDSL.UFA)
library(mzR)
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),"ufa_trace_lib.R"))
if (is.null(config$profile_manifest)) stop("Build the shared native profile cache before study scoring")
manifest <- jsonlite::read_json(config$profile_manifest,simplifyVector=FALSE)
stopifnot(identical(manifest$spec$engine,as.character(packageVersion("IDSL.UFA"))),
          identical(manifest$spec$IPA,as.character(packageVersion("IDSL.IPA"))))
databases <- list()
needed <- unique(vapply(config$queries,function(q)q$adduct,character(1)))
for (adduct in intersect(names(manifest$adducts),needed)) {
  entry <- manifest$adducts[[adduct]]
  stopifnot(digest::digest(file=entry$path,algo="sha256")==entry$sha256)
  record <- readRDS(entry$path)
  stopifnot(identical(record$spec,ufa_cache_spec()))
  databases[[adduct]] <- record$db
}
output <- args[2]
if (dir.exists(output)) stop("Preserve earlier native attempts")
dir.create(output,recursive=TRUE)
start <- proc.time()[[3]]
ms <- openMSfile(config$sample$raw_path)
header <- header(ms)
ms1 <- which(header$msLevel==1)
spectra <- peaks(ms,ms1)
stopifnot(length(spectra)==length(ms1))
times <- header$retentionTime[ms1]/60
aligned <- approx(unlist(config$sample$raw_rt_min),unlist(config$sample$corrected_rt_min),
                  xout=times,rule=1)$y
close(ms)
connection <- file(file.path(output,"predictions.jsonl"),"wt")
lossless_checks <- 0L
for (q in config$queries) {
  print(paste(config$stream,q$query_id));flush.console()
  prediction <- list(query_id=q$query_id,stream=q$stream,feature_id=q$feature_id,
                      status=q$status,candidates=list(),generated_N=q$generated_N)
  if (q$status=="ready" && length(q$candidates)>0) {
    prediction <- tryCatch({
      chosen <- which(is.finite(aligned) & aligned>=q$boundaries$rt_start_corrected_min &
                      aligned<=q$boundaries$rt_end_corrected_min)
      if (length(chosen)<2) stop("Insufficient native scan coverage")
      window <- spectra[chosen]
      # The observed anchor is chosen independently of candidate formulas/ranks.
      anchor <- ufa_observed_anchor(window,q$mz,q$charge,times=aligned[chosen])
      db <- ufa_subset_database(databases[[q$adduct]],vapply(q$candidates,function(c)c$ion_formula,character(1)))
      result <- ufa_annotate(db,window,anchor,q$rt)
      if (lossless_checks<4L) {
        if (!identical(result,ufa_annotate(db,window,anchor,q$rt,prune=FALSE)))
          stop(structure(list(message="Native lossless-pruning equivalence failed",call=NULL),
                         class=c("ufa_parity_error","error","condition")))
        lossless_checks <- lossless_checks+1L
      }
      candidates <- ufa_native_candidates(result,q)
      list(query_id=q$query_id,stream=q$stream,feature_id=q$feature_id,status="complete",candidates=candidates,
           generated_N=q$generated_N,native_DB_N=length(db$MolecularFormula),scan_N=length(window),
           observed_anchor=anchor,profile_cache_reused=TRUE,lossless_native_window_pruning=TRUE)
    },error=function(e) {
      if(inherits(e,"ufa_parity_error"))stop(e)
      prediction$status <- paste("unavailable:",conditionMessage(e));prediction
    })
  } else if (q$status=="ready") prediction$status <- "empty generated candidate pool"
  writeLines(jsonlite::toJSON(prediction,auto_unbox=TRUE,null="null",na="null",digits=NA),connection)
  flush(connection)
}
close(connection)
writeLines(capture.output(sessionInfo()),file.path(output,"sessionInfo.txt"))
jsonlite::write_json(list(engine="IDSL.UFA",version=as.character(packageVersion("IDSL.UFA")),
    stream=config$stream,queries=length(config$queries),seconds=proc.time()[[3]]-start,
    known_adduct_conditional=TRUE,shared_unranked_candidate_universe=TRUE,native_IPA_peak_detection=FALSE,
    bridge_version=UFA_BRIDGE_VERSION,anchor_settings=UFA_ANCHOR_SETTINGS,
    actual_query_lossless_checks=lossless_checks,profile_cache_sha256=digest::digest(file=config$profile_manifest,algo="sha256")),
    file.path(output,"receipt.json"),auto_unbox=TRUE,pretty=TRUE)
