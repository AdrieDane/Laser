# Complete, resumable native profile preparation. No observed spectra or labels.
# Rscript build_ufa_profile_cache.R CONFIG.json NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
config <- jsonlite::read_json(args[1],simplifyVector=FALSE)
.libPaths(c(config$r_library,.libPaths()));library(IDSL.UFA)
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
helper <- file.path(dirname(normalizePath(script)),"ufa_trace_lib.R");source(helper)
output <- args[2];if (dir.exists(output)) stop("Preserve previous cache builds")
dir.create(output,recursive=TRUE)
began <- proc.time()[[3]];spec <- ufa_cache_spec()
root <- if(is.null(config$cache_root))file.path(output,"profiles") else config$cache_root
workers <- if(is.null(config$workers))1L else as.integer(config$workers)
stopifnot(workers>=1L,workers<=4L)
by_adduct <- list()
for (path in unlist(config$mode_configs)) {
  mode <- jsonlite::read_json(path,simplifyVector=FALSE)
  for (q in mode$queries) {
    if (!is.null(config$adduct_subset) && !q$adduct%in%unlist(config$adduct_subset)) next
    # All generated candidates, including queries with unavailable raw peaks.
    pool <- vapply(q$candidates,function(c)c$formula,character(1))
    by_adduct[[q$adduct]][[length(by_adduct[[q$adduct]])+1L]] <- pool
  }
}
by_adduct <- lapply(by_adduct,function(x)sort(unique(unlist(x,use.names=FALSE))))
stopifnot(length(by_adduct)>0L)
plans <- lapply(sort(names(by_adduct)),function(adduct) {
  path <- file.path(root,digest::digest(adduct,algo="sha256",serialize=FALSE))
  ufa_cache_plan(by_adduct[[adduct]],adduct,path,spec)
})
# Check committed complete records first; only genuinely missing batches need work.
tasks <- unlist(lapply(plans,function(p) {
  if (!is.null(ufa_read_record(p$path,p$key,p$spec))) return(list())
  p$chunks
}),recursive=FALSE)
cat("Native preparation:",sum(vapply(plans,function(p)length(p$formulas),integer(1))),
    "formula/adduct pairs;",length(tasks),"batches;",workers,"workers\n");flush.console()
if (workers==1L) receipts <- lapply(tasks,ufa_prepare_chunk) else {
  cluster <- parallel::makePSOCKcluster(workers,outfile=file.path(output,"workers.log"))
  tryCatch({
    parallel::clusterCall(cluster,function(library_path,helper_path,expected) {
      .libPaths(c(library_path,.libPaths()));library(IDSL.UFA);source(helper_path,.GlobalEnv)
      stopifnot(identical(ufa_cache_spec(),expected));TRUE
    },config$r_library,helper,spec)
    receipts <- parallel::parLapplyLB(cluster,tasks,function(task) {
      answer <- ufa_prepare_chunk(task)
      if (task$batch%%8L==0L) {cat(task$adduct,"batch",task$batch,
        if(answer$reused)"reused" else "committed","\n");flush.console()}
      answer
    },chunk.size=1L)
  },finally=parallel::stopCluster(cluster))
}
manifest <- list(schema=2L,spec=spec,adducts=list(),workers=workers,
  built_chunks=sum(!vapply(receipts,function(x)x$reused,logical(1))),
  reused_chunks=sum(vapply(receipts,function(x)x$reused,logical(1))))
for (plan in plans) {
  cache <- ufa_cached_database(plan$formulas,plan$adduct,dirname(plan$path),merge_only=TRUE)
  probes <- plan$formulas[unique(round(seq(1,length(plan$formulas),length.out=min(8L,length(plan$formulas)))))]
  reference <- IDSL.UFA::molecularFormula2IPdb(probes,IonPathways=plan$adduct,peak_spacing=.005,
      intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  stopifnot(identical(reference,ufa_subset_database(cache$db,reference$MolecularFormula)))
  manifest$adducts[[plan$adduct]] <- cache[setdiff(names(cache),"db")]
  manifest$adducts[[plan$adduct]]$requested_formulas <- length(plan$formulas)
  manifest$adducts[[plan$adduct]]$native_profiles <- length(cache$db$MolecularFormula)
  manifest$adducts[[plan$adduct]]$unmodified_native_probe_parity <- length(probes)
}
stopifnot(identical(spec,ufa_cache_spec()))
manifest$seconds <- proc.time()[[3]]-began
jsonlite::write_json(manifest,file.path(output,"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
cat("ALL NATIVE PROFILE CACHES COMPLETE\n")
