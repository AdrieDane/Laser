# Native conditional IDSL.UFA reranking; shared candidate generation is external.
# Rscript native_ufa_trace.R PRIVATE_CONFIG.json NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
config <- jsonlite::read_json(args[1],simplifyVector=FALSE)
.libPaths(c(config$r_library,.libPaths()))
library(IDSL.UFA)
library(mzR)
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
      raw <- lapply(window,function(s) {
        index <- which(abs(s[,1]-q$mz)<=q$mz*4e-6)
        if (!length(index)) return(c(0,0))
        c(s[index[which.min(abs(s[index,1]-q$mz))],2],length(index))
      })
      raw <- do.call(rbind,raw)
      height <- max(raw[,1])
      if (height<=0) stop("No observed target centroid")
      if (any(raw[raw[,1]>=.2*height,2]>1)) stop("Ambiguous target mass centroids")
      z <- abs(q$charge)
      isotope <- vapply(window,function(s) {
        index <- which(abs(s[,1]-q$mz-1.003354835336/z)<=(q$mz+1.003354835336/z)*4e-6)
        if (!length(index)) return(0)
        s[index[which.min(abs(s[index,1]-q$mz-1.003354835336/z))],2]
      },numeric(1))
      pl <- matrix(0,nrow=1,ncol=11)
      pl[1,c(1,2,3,4,8,11)] <- c(1,length(window),q$rt,height,q$mz,100*sum(isotope)/sum(raw[,1]))
      pool <- vapply(q$candidates,function(c)c$formula,character(1))
      db <- molecularFormula2IPdb(pool,IonPathways=q$adduct,peak_spacing=.005,
          intensity_cutoff_str="min(c(c, 10))",number_processing_threads=1,allowedVerbose=FALSE)
      # This bridge supplies original monoisotopic feature coordinates. Native
      # UFA anchors on the most abundant isotope. Do not silently score a mixed
      # candidate pool whose anchor definitions differ (common for high C).
      if (any(db$IndexMAIso!=1L)) {
        stop("Unsupported most-abundant-isotope anchor; original-monoisotope bridge is not validated for this candidate pool")
      }
      result <- molecular_formula_annotator(db,window,pl,1,massAccuracy=.01,maxNEME=5,
          minPCS=950,minNDCS=2,minRCS=5,scoreCoefficients=rep(1,5),number_processing_threads=1)
      candidates <- list()
      if (!is.null(result)) {
        for (i in seq_len(nrow(result))) {
          neutral <- q$candidates[vapply(q$candidates,function(c)c$ion_formula==result$IonFormula[i],logical(1))]
          if (length(neutral)!=1L) stop("Native ion composition has an ambiguous neutral-formula crosswalk")
          candidates[[i]] <- list(formula=neutral[[1]]$formula,adduct=q$adduct,charge=q$charge,
            rank=as.integer(result$rank[i]),native_ion_formula=result$IonFormula[i],
            PCS=as.numeric(result$PCS[i]),NEME=as.numeric(result$normEucMassError[i]),
            NDCS=as.integer(result$NDCS[i]),RCS=as.numeric(result$`ratioChromScan(%)`[i]))
        }
      }
      list(query_id=q$query_id,stream=q$stream,feature_id=q$feature_id,status="complete",candidates=candidates,
           generated_N=q$generated_N,native_DB_N=length(db$MolecularFormula),scan_N=length(window),actual_observed_apex_height=height)
    },error=function(e) {prediction$status <- paste("unavailable:",conditionMessage(e));prediction})
  } else if (q$status=="ready") prediction$status <- "empty generated candidate pool"
  writeLines(jsonlite::toJSON(prediction,auto_unbox=TRUE,null="null",na="null",digits=NA),connection)
  flush(connection)
}
close(connection)
writeLines(capture.output(sessionInfo()),file.path(output,"sessionInfo.txt"))
jsonlite::write_json(list(engine="IDSL.UFA",version=as.character(packageVersion("IDSL.UFA")),
    stream=config$stream,queries=length(config$queries),seconds=proc.time()[[3]]-start,
    known_adduct_conditional=TRUE,shared_unranked_candidate_universe=TRUE,native_IPA_peak_detection=FALSE),
    file.path(output,"receipt.json"),auto_unbox=TRUE,pretty=TRUE)
