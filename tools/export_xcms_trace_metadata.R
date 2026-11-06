# Read-only legacy XCMS bridge. No annotation, formula or significance labels.
# Rscript export_xcms_trace_metadata.R PROJECT RESULTS_ROOT NEW_OUTPUT
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L, requireNamespace("jsonlite", quietly = TRUE))
project <- normalizePath(args[1], mustWork = TRUE)
results <- normalizePath(args[2], mustWork = TRUE)
output <- args[3]
if (dir.exists(output)) stop("Preserve an existing export; choose a new output directory")
dir.create(output, recursive = TRUE)
options(digits = 17)
streams <- c("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
streams <- streams[file.exists(file.path(project, "XCMS", paste0("peakgrouplist_", streams, ".txt")))]
if (!length(streams)) stop("No canonical XCMS peak-group inputs found")
manifest <- list(schema_version = 1L, peak_rt_basis = "corrected",
                 provenance = "Original xcmsSet groupidx, corrected peak boundaries and per-injection RT maps",
                 feature_scope = "all original XCMS groups", streams = list())
for (stream in streams) {
  source <- file.path(results, stream, paste0("xsetfinal_only_", stream, ".Rdata"))
  if (!file.exists(source)) stop("Original XCMS object missing: ", source)
  environment <- new.env(parent = baseenv())
  objects <- load(source, envir = environment)
  candidates <- objects[vapply(objects, function(n) "xcmsSet" %in% as.character(attr(get(n, environment), "class")), logical(1))]
  if (length(candidates) != 1L) stop("Expected exactly one xcmsSet")
  x <- get(candidates[1], environment)
  groups <- attr(x, "groups"); peaks <- attr(x, "peaks")
  indices <- attr(x, "groupidx"); times <- attr(x, "rt")
  paths <- attr(x, "filepaths")
  if (is.null(times$corrected) || length(times$raw) != length(paths)) stop("Missing original RT alignment")
  input <- file.path(project, "XCMS", paste0("peakgrouplist_", stream, ".txt"))
  header <- strsplit(readLines(input, n = 1), "\t", fixed = TRUE)[[1]]
  columns <- rep("NULL", length(header)); columns[1:7] <- "character"
  frame <- read.delim(input, check.names = FALSE, colClasses = columns)
  stopifnot(nrow(frame) == nrow(groups), length(indices) == nrow(groups), !anyDuplicated(frame$name))
  dm <- abs(as.numeric(frame$mzmed) - groups[, "mzmed"])
  dt <- abs(as.numeric(frame$rtmed) - groups[, "rtmed"] / 60)
  # Export rounding only; this is a coordinate crosswalk, never an annotation tolerance.
  if (any(dm > .0001 | dt > .0001)) stop("XCMS group order/coordinates differ from exported original features")
  files <- list.files(file.path(project, "Raw", stream), recursive = TRUE, full.names = TRUE,
                      pattern = "\\.(mzXML|mzML)$", ignore.case = TRUE)
  stems <- tolower(sub("\\.(mzXML|mzML)$", "", basename(files), ignore.case = TRUE))
  if (anyDuplicated(stems)) stop("Ambiguous raw file stems in ", stream)
  samples <- sub("\\.(mzXML|mzML)$", "", basename(gsub("\\\\", "/", paths)), ignore.case = TRUE)
  raw_paths <- files[match(tolower(samples), stems)]
  stopifnot(!anyNA(raw_paths), !anyDuplicated(samples))
  sample_maps <- lapply(seq_along(paths), function(i) {
    a <- as.numeric(times$raw[[i]]) / 60; b <- as.numeric(times$corrected[[i]]) / 60
    stopifnot(length(a) == length(b), all(is.finite(a)), all(is.finite(b)), all(diff(a) > 0), all(diff(b) > 0))
    list(sample = samples[i], raw_path = substring(normalizePath(raw_paths[i]), nchar(project) + 2),
         raw_rt_min = a, corrected_rt_min = b)
  })
  peak_rows <- unlist(indices, use.names = FALSE)
  group_rows <- rep(seq_along(indices), lengths(indices))
  stopifnot(all(peak_rows >= 1L & peak_rows <= nrow(peaks)))
  selected <- peaks[peak_rows, , drop = FALSE]
  sample_index <- as.integer(selected[, "sample"])
  stopifnot(all(sample_index >= 1L & sample_index <= length(samples)))
  boundaries <- data.frame(stream = stream, feature_id = as.character(frame$name[group_rows]),
    sample = samples[sample_index], source_peak_row = peak_rows,
    target_mz = as.numeric(groups[group_rows, "mzmed"]),
    rt_start_corrected_min = selected[, "rtmin"] / 60,
    rt_apex_corrected_min = selected[, "rt"] / 60,
    rt_end_corrected_min = selected[, "rtmax"] / 60,
    filled = peak_rows %in% attr(x, "filled"), stringsAsFactors = FALSE)
  destination <- file.path(output, paste0(stream, "_boundaries.csv.gz"))
  connection <- gzfile(destination, "wt")
  write.csv(boundaries, connection, row.names = FALSE, na = "")
  close(connection)
  manifest$streams[[stream]] <- list(native_source = source, original_groups = nrow(groups),
    native_source_md5 = unname(tools::md5sum(source)),
    input_table = substring(normalizePath(input), nchar(project) + 2),
    input_md5 = unname(tools::md5sum(input)),
    coordinate_max_mass_delta = max(dm), coordinate_max_rt_delta_min = max(dt),
    boundaries = basename(destination), samples = sample_maps)
  cat("Exported", stream, nrow(groups), "original groups,", nrow(boundaries), "peak memberships\n")
  rm(x, environment, groups, peaks, selected, boundaries); gc()
}
jsonlite::write_json(manifest, file.path(output, "manifest.json"), auto_unbox = TRUE,
                     pretty = FALSE, digits = NA, null = "null")
cat("Read-only XCMS trace metadata export complete\n")
