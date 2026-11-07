# Actual killed-writer, resume, corruption and parallel/native equivalence tests.
# Rscript selftest_ufa_cache.R R_LIBRARY NEW_OUTPUT [interrupt_child]
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)%in%c(2L,3L))
.libPaths(c(args[1],.libPaths()));library(IDSL.UFA)
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
helper <- file.path(dirname(normalizePath(script)),"ufa_trace_lib.R");source(helper)
out <- args[2]
grid <- expand.grid(C=seq(24,120,2),O=c(4,8,12,16),N=c(0,1),P=c(0,1))
grid <- grid[unique(round(seq(1,nrow(grid),length.out=768))),]
pool <- sprintf("C%dH%d%sO%d%s",grid$C,2*grid$C-2,
    ifelse(grid$N>0,paste0("N",grid$N),""),grid$O,ifelse(grid$P>0,"P",""))
if(length(args)==3L) {
  plan <- ufa_cache_plan(pool,"[M+H]+",file.path(out,"interrupted"))
  ufa_prepare_chunk(plan$chunks[[1]])
  ufa_with_lock(plan$chunks[[2]]$path,{tools::pskill(Sys.getpid(),9L);Sys.sleep(2)})
  stop("Killed-writer fixture did not stop")
}
if(dir.exists(out))stop("Preserve old self-tests");dir.create(out,recursive=TRUE)
began <- proc.time()[[3]];before <- ufa_cache_spec()
status <- system2(file.path(R.home("bin"),"Rscript"),c("--vanilla",shQuote(normalizePath(script)),
    shQuote(args[1]),shQuote(out),"interrupt_child"),stdout=file.path(out,"child.log"),stderr=file.path(out,"child.log"))
stopifnot(status!=0)
plan <- ufa_cache_plan(pool,"[M+H]+",file.path(out,"interrupted"))
first_sha <- digest::digest(file=plan$chunks[[1]]$path,algo="sha256")
first_time <- file.info(plan$chunks[[1]]$path)$mtime
stopifnot(dir.exists(paste0(plan$chunks[[2]]$path,".lock")))
missing_merge <- tryCatch(ufa_cached_database(pool,"[M+H]+",dirname(plan$path),merge_only=TRUE),error=function(e)e)
stopifnot(inherits(missing_merge,"error"))
resumed <- ufa_cached_database(pool,"[M+H]+",dirname(plan$path))
stopifnot(resumed$built_chunks==2L,resumed$reused_chunks==1L,
          digest::digest(file=plan$chunks[[1]]$path,algo="sha256")==first_sha,
          identical(file.info(plan$chunks[[1]]$path)$mtime,first_time),
          !dir.exists(paste0(plan$chunks[[2]]$path,".lock")))
cold <- ufa_cached_database(pool,"[M+H]+",file.path(out,"cold"))
warm <- ufa_cached_database(pool,"[M+H]+",file.path(out,"cold"))
stopifnot(identical(resumed$db,cold$db),identical(cold$db,warm$db),warm$cache_hit,
          identical(first_sha,digest::digest(file=plan$chunks[[1]]$path,algo="sha256")))
# An interrupted data file with no commit checksum is regenerated, not trusted.
orphan <- ufa_cache_plan(pool[1:8],"[M+H]+",file.path(out,"orphan"))$chunks[[1]]
writeLines("unfinished",orphan$path);ufa_prepare_chunk(orphan)
stopifnot(!is.null(ufa_read_record(orphan$path,orphan$key,orphan$spec)))
# Reject a committed corrupt record; never silently rebuild it.
con <- file(orphan$path,"ab");writeBin(as.raw(1),con);close(con)
corrupt <- tryCatch(ufa_prepare_chunk(orphan),error=function(e)e)
stopifnot(inherits(corrupt,"error"),grepl("Corrupt",conditionMessage(corrupt)))
# A live writer cannot be displaced by a second caller.
live <- file.path(out,"live.rds")
busy <- ufa_with_lock(live,tryCatch(ufa_with_lock(live,TRUE),error=function(e)e))
stopifnot(inherits(busy,"error"),grepl("in use",conditionMessage(busy)))
# Four workers, mixed adducts; every generated native profile must equal the
# unmodified engine (not only the selected top-scoring formula).
adducts <- c("[M+H]+","[M+NH4]+","[M-H]-","[M-2H]2-")
queries <- lapply(adducts,function(a)list(adduct=a,candidates=lapply(pool,function(f)list(formula=f))))
input <- file.path(out,"input.json");jsonlite::write_json(list(queries=queries),input,auto_unbox=TRUE)
config <- file.path(out,"config.json");jsonlite::write_json(list(r_library=args[1],mode_configs=list(input),workers=4L),config,auto_unbox=TRUE)
code <- system2(file.path(R.home("bin"),"Rscript"),c("--vanilla",shQuote(file.path(dirname(script),"build_ufa_profile_cache.R")),shQuote(config),shQuote(file.path(out,"parallel"))),stdout=file.path(out,"parallel.log"),stderr=file.path(out,"parallel.log"))
stopifnot(code==0L)
manifest <- jsonlite::read_json(file.path(out,"parallel","manifest.json"),simplifyVector=FALSE)
for(adduct in adducts) {
  record <- readRDS(manifest$adducts[[adduct]]$path)
  serial <- ufa_cached_database(pool,adduct,file.path(out,paste0("serial-",match(adduct,adducts))))
  stopifnot(identical(record$db,serial$db))
  probes <- sort(unique(pool))[unique(round(seq(1,length(pool),length.out=64L)))]
  native <- IDSL.UFA::molecularFormula2IPdb(probes,IonPathways=adduct,
      peak_spacing=.005,intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
  stopifnot(identical(ufa_subset_database(record$db,native$MolecularFormula),native))
}
fractional <- data.frame(IonFormula=c("C2H5","C3H7"),rank=c(1.5,1.5),PCS=1000,
                        normEucMassError=0,NDCS=5,check.names=FALSE)
fractional$`ratioChromScan(%)` <- 100
query <- list(adduct="[M+H]+",charge=1,candidates=list(
  list(formula="C2H4",ion_formula="C2H5"),list(formula="C3H6",ion_formula="C3H7")))
stopifnot(identical(vapply(ufa_native_candidates(fractional,query),function(c)c$rank,numeric(1)),c(1.5,1.5)))
stopifnot(identical(before,ufa_cache_spec()))
jsonlite::write_json(list(passed=TRUE,actual_killed_writer=TRUE,completed_batch_unchanged=TRUE,
  resumed_built_chunks=resumed$built_chunks,resumed_reused_chunks=resumed$reused_chunks,
  cold_warm_parallel_native_exact=TRUE,unmodified_native_profile_parity_N=length(probes)*4L,
  full_parallel_serial_profile_parity_N=length(pool)*4L,fractional_native_ranks_preserved=TRUE,
  incomplete_merge_rejected=TRUE,uncommitted_orphan_rebuilt=TRUE,corrupt_commit_rejected=TRUE,
  live_lock_preserved=TRUE,cache_descriptor_stable=TRUE,seconds=proc.time()[[3]]-began,
  study_data_used=FALSE,spec=before),file.path(out,"receipt.json"),auto_unbox=TRUE,pretty=TRUE)
cat("RESUMABLE CACHE CONTROLS PASSED\n")
