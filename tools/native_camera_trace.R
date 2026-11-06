# Native configured CAMERA comparison; never loads curator/reference labels.
# Rscript native_camera_trace.R PRIVATE_CONFIG.json NEW_OUTPUT
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
config <- jsonlite::read_json(args[1], simplifyVector = TRUE)
.libPaths(c(config$r_library, .libPaths()))
library(CAMERA)
output <- args[2]
if (dir.exists(output)) stop("Preserve earlier native attempts")
dir.create(output, recursive = TRUE)
start <- proc.time()[[3]]
native <- new.env(parent = baseenv())
objects <- load(config$native_source, native)
names <- objects[vapply(objects, function(n) "xcmsSet" %in% as.character(attr(get(n,native),"class")), logical(1))]
stopifnot(length(names) == 1L)
x <- get(names[1],native)
stems <- function(p) tolower(sub("\\.(mzXML|mzML)$", "", basename(gsub("\\\\", "/", p)),ignore.case=TRUE))
paths <- config$samples$raw_path[match(stems(x@filepaths),stems(config$samples$raw_path))]
stopifnot(!anyNA(paths), !anyDuplicated(paths))
x@filepaths <- paths
sample <- which(stems(paths) == tolower(config$sample))
stopifnot(length(sample) == 1L)
input <- read.delim(config$input_table, check.names=FALSE)
stopifnot(nrow(input)==nrow(x@groups),max(abs(input$mzmed-x@groups[,"mzmed"]))<=.0001,
          max(abs(input$rtmed-x@groups[,"rtmed"]/60))<=.0001)
polarity <- if (grepl("Neg$",config$stream)) "negative" else "positive"
if (!is.null(config$resume_group_corr)) {
  stopifnot(unname(tools::md5sum(config$resume_group_corr))==config$resume_md5)
  annotation <- readRDS(config$resume_group_corr)
} else {
  annotation <- xsAnnotate(x, sample=sample,nSlaves=1,polarity=polarity)
  annotation <- groupFWHM(annotation,sigma=6,perfwhm=.6,intval="maxo")
  saveRDS(annotation,file.path(output,"after_FWHM.rds"))
  annotation <- findIsotopes(annotation,maxcharge=2,maxiso=4,ppm=4,mzabs=0,intval="maxo",minfrac=.5)
  annotation <- groupCorr(annotation,cor_eic_th=.85,pval=.05,graphMethod="hcs",
                          calcIso=FALSE,calcCiS=TRUE,calcCaS=FALSE)
}
saveRDS(annotation,file.path(output,"after_groupCorr.rds"))
annotation <- findAdducts(annotation,ppm=4,mzabs=0,multiplier=3,polarity=polarity,
                         rules=read.csv(config$rules,stringsAsFactors=FALSE),max_peaks=100)
peaks <- getPeaklist(annotation)
stopifnot(nrow(peaks)==nrow(input),max(abs(peaks$mz-x@groups[,"mzmed"]))<1e-9)
peaks$feature_id <- as.character(input$name)
peaks$stream <- config$stream
write.csv(peaks,file.path(output,"native_peaklist.csv"),row.names=FALSE,na="")
saveRDS(annotation,file.path(output,"annotation.rds"))
writeLines(capture.output(sessionInfo()),file.path(output,"sessionInfo.txt"))
jsonlite::write_json(list(engine="CAMERA",version=as.character(packageVersion("CAMERA")),
    stream=config$stream,original_groups=nrow(input),seconds=proc.time()[[3]]-start,
    first_qc=config$sample,configured_rules=TRUE,across_sample_height_correlation=FALSE,
    same_injection_shape_correlation=TRUE),file.path(output,"receipt.json"),auto_unbox=TRUE,pretty=TRUE)
