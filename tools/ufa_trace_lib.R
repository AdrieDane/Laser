# Native IDSL.UFA bridge helpers. No reference identities enter these functions.
UFA_BRIDGE_VERSION <- 2L
UFA_ANCHOR_SETTINGS <- list(ppm=4, max_order=4L, min_points=5L, edges=2L,
                           relative_height=.1, shape_r=.85, overlap=.5, tie_fraction=.05)

ufa_exact_generator <- function() {
  # Clone, never modify the installed namespace. Retain native elemental
  # probabilities, cutoff, peak merging and rounding. Memoize pure elemental
  # results and enumerate Cartesian sums/products in the same last-index-fast
  # order and left-associative arithmetic as the native recursive traversal.
  generator <- IDSL.UFA::molecularFormula2IPdb
  profile <- get("isotopic_profile_calculator",asNamespace("IDSL.UFA"))
  env <- new.env(parent=environment(profile))
  cache <- new.env(parent=emptyenv())
  env$.ufa_memoize <- function(original) {
    force(original)
    function(iMassAbundanceList,numberAtoms,UFA_IP_memeory_variables2) {
      key <- digest::digest(list(iMassAbundanceList,numberAtoms,UFA_IP_memeory_variables2))
      if (exists(key,cache,inherits=FALSE)) return(get(key,cache,inherits=FALSE))
      result <- original(iMassAbundanceList,numberAtoms,UFA_IP_memeory_variables2)
      assign(key,result,cache);result
    }
  }
  env$.ufa_states <- function(mass_parts,probability_parts,cutoff) {
    mass <- 0;probability <- 1
    for (i in seq_along(mass_parts)) {
      x <- mass_parts[[i]];p <- probability_parts[[i]]
      stopifnot(length(x)==length(p),all(is.finite(p)),all(p>=0 & p<=1))
      n <- length(mass)
      mass <- rep(mass,each=length(x))+rep(x,times=n)
      probability <- rep(probability,each=length(p))*rep(p,times=n)
      # Every subsequent factor is in [0,1]. A branch already <= the native
      # strict final cutoff can never survive it. Retained states have exactly
      # the same arithmetic and relative enumeration order as native recursion.
      keep <- probability>cutoff
      mass <- mass[keep];probability <- probability[keep]
    }
    list(mass=mass,probability=probability)
  }
  found <- c(element=0L,mass=0L,abundance=0L,mass_allocation=0L,abundance_allocation=0L)
  transform <- function(x) {
    if (!is.call(x)) return(x)
    if (identical(x[[1]],as.name("function"))) return(x)
    # These native work arrays are overwritten by the exact Cartesian result
    # before any read. Avoid allocating the unpruned product of isotope counts.
    if (found['mass']==0L && identical(x,quote(MZ <- rep(0,nIsotopicCombinations)))) {
      found['mass_allocation'] <<- found['mass_allocation']+1L
      return(quote(MZ <- numeric(0)))
    }
    if (identical(x,quote(RA <- rep(1,nIsotopicCombinations)))) {
      found['abundance_allocation'] <<- found['abundance_allocation']+1L
      return(quote(RA <- numeric(0)))
    }
    if (identical(x[[1]],as.name("<-")) && identical(x[[2]],as.name("SUB_COMB"))) {
      found['element'] <<- found['element']+1L
      return(substitute({ORIGINAL;SUB_COMB <- .ufa_memoize(SUB_COMB)},list(ORIGINAL=x)))
    }
    if (identical(x[[1]],as.name("RecursiveMass"))) {
      found['mass'] <<- found['mass']+1L
      return(quote({.ufa_state <- .ufa_states(massCombinations,abundanceCombinations,UFA_IP_memeory_variables[2]);MZ <- .ufa_state$mass}))
    }
    if (identical(x[[1]],as.name("RecursiveAbundance"))) {
      found['abundance'] <<- found['abundance']+1L
      return(quote(RA <- .ufa_state$probability))
    }
    for (i in seq_along(x)) if (i>1L) x[[i]] <- transform(x[[i]])
    x
  }
  body(profile) <- transform(body(profile))
  if (any(found!=1L)) stop(paste("Native calculator structure changed:",paste(names(found),found,collapse=", ")))
  environment(profile) <- env
  parent <- new.env(parent=environment(generator))
  parent$isotopic_profile_calculator <- profile
  # Native element_sorter reloads the same constants file on every batch.
  # Memoize its pure result by both arguments in cloned environments only.
  sorter <- get("element_sorter",asNamespace("IDSL.UFA"))
  sorted_elements <- new.env(parent=emptyenv())
  parent$element_sorter <- function(ElementList="all",alphabeticalOrder=TRUE) {
    key <- digest::digest(list(ElementList,alphabeticalOrder))
    if (!exists(key,sorted_elements,inherits=FALSE))
      assign(key,sorter(ElementList,alphabeticalOrder),sorted_elements)
    get(key,sorted_elements,inherits=FALSE)
  }
  filter <- get("molecular_formula_elements_filter",asNamespace("IDSL.UFA"))
  environment(filter) <- parent
  parent$molecular_formula_elements_filter <- filter
  environment(generator) <- parent
  generator
}

UFA_EXACT_GENERATOR <- NULL
UFA_GENERATOR_SPEC <- NULL

ufa_cache_spec <- function() {
  list(schema=3L, R=as.character(getRversion()), engine=as.character(packageVersion("IDSL.UFA")),
       IPA=as.character(packageVersion("IDSL.IPA")), peak_spacing=.005,
       native_isotope_constants=digest::digest(file=system.file("extdata","IUPAC_Isotopes.rda",package="IDSL.UFA",mustWork=TRUE),algo="sha256"),
       intensity_cutoff="min(c(c, 10))",optimizer="exact-element-memo-cartesian-tail-no-prealloc-v3",
       optimizer_code=digest::digest(paste(deparse(body(ufa_exact_generator),width.cutoff=500L),collapse="\n"),algo="sha256",serialize=FALSE),batch_size=256L,
       generator=digest::digest(paste(deparse(body(IDSL.UFA::molecularFormula2IPdb),width.cutoff=500L),collapse="\n"),algo="sha256",serialize=FALSE),
       native_profile=digest::digest(paste(deparse(body(get("isotopic_profile_calculator",asNamespace("IDSL.UFA"))),width.cutoff=500L),collapse="\n"),algo="sha256",serialize=FALSE))
}

ufa_atomic_record <- function(record,path) {
  temporary <- paste0(path,".tmp-",Sys.getpid())
  on.exit(unlink(c(temporary,paste0(temporary,".sha256"))))
  saveRDS(record,temporary,compress=FALSE)
  sha <- digest::digest(file=temporary,algo="sha256")
  writeLines(sha,paste0(temporary,".sha256"))
  stopifnot(file.rename(temporary,path),file.rename(paste0(temporary,".sha256"),paste0(path,".sha256")))
  sha
}

ufa_read_record <- function(path,key,spec) {
  # A data file without its commit checksum was interrupted; never trust it.
  if (!file.exists(path) || !file.exists(paste0(path,".sha256"))) return(NULL)
  sha <- readLines(paste0(path,".sha256"),warn=FALSE)
  if (length(sha)!=1L || digest::digest(file=path,algo="sha256")!=sha)
    stop(paste("Corrupt native cache record:",path))
  record <- readRDS(path)
  stopifnot(identical(record$key,key),identical(record$spec,spec))
  record
}

ufa_with_lock <- function(path,code) {
  lock <- paste0(path,".lock")
  token <- paste(Sys.info()[['nodename']],Sys.getpid(),format(Sys.time(),"%OS6"),sep=":")
  repeat {
    if (dir.create(lock,showWarnings=FALSE)) {
      saveRDS(list(pid=Sys.getpid(),host=Sys.info()[['nodename']],token=token),file.path(lock,"owner.rds"))
      break
    }
    owner <- tryCatch(readRDS(file.path(lock,"owner.rds")),error=function(e)NULL)
    if (!is.null(owner) && identical(owner$host,Sys.info()[['nodename']]) &&
        !isTRUE(suppressWarnings(tools::pskill(owner$pid,0L)))) {
      abandoned <- paste0(lock,".abandoned-",Sys.getpid())
      if (file.rename(lock,abandoned)) unlink(abandoned,recursive=TRUE)
    } else stop(paste("Native cache record is in use (retry after its writer stops):",path))
  }
  on.exit({
    owner <- tryCatch(readRDS(file.path(lock,"owner.rds")),error=function(e)NULL)
    if (!is.null(owner) && identical(owner$token,token)) unlink(lock,recursive=TRUE)
  })
  force(code)
}

ufa_cache_plan <- function(formulas,adduct,directory,spec=ufa_cache_spec()) {
  formulas <- sort(unique(formulas));stopifnot(length(formulas)>0)
  dir.create(directory,recursive=TRUE,showWarnings=FALSE)
  key <- digest::digest(list(spec,formulas,adduct),algo="sha256")
  batches <- split(formulas,ceiling(seq_along(formulas)/spec$batch_size))
  chunks <- lapply(seq_along(batches),function(i) {
    pool <- unname(batches[[i]])
    chunk_key <- digest::digest(list(spec,pool,adduct),algo="sha256")
    list(key=chunk_key,spec=spec,formulas=pool,adduct=adduct,
         path=file.path(directory,paste0("batch-",chunk_key,".rds")),batch=i)
  })
  list(key=key,spec=spec,formulas=formulas,adduct=adduct,
       path=file.path(directory,paste0(key,".rds")),chunks=chunks)
}

ufa_prepare_chunk <- function(task) {
  record <- ufa_read_record(task$path,task$key,task$spec)
  reused <- !is.null(record)
  if (!reused) record <- ufa_with_lock(task$path,{
    existing <- ufa_read_record(task$path,task$key,task$spec)
    if (!is.null(existing)) existing else {
      if (is.null(UFA_EXACT_GENERATOR) || !identical(UFA_GENERATOR_SPEC,task$spec)) {
        UFA_EXACT_GENERATOR <<- ufa_exact_generator()
        UFA_GENERATOR_SPEC <<- task$spec
      }
      db <- UFA_EXACT_GENERATOR(task$formulas,IonPathways=task$adduct,
          peak_spacing=task$spec$peak_spacing,intensity_cutoff_str=task$spec$intensity_cutoff,
          number_processing_threads=1,allowedVerbose=FALSE)
      if (!is.list(db$IsotopicProfile)) stop("Native profile batch has no usable profiles")
      value <- list(key=task$key,spec=task$spec,adduct=task$adduct,formulas=task$formulas,db=db)
      ufa_atomic_record(value,task$path)
      value
    }
  })
  stopifnot(identical(record$formulas,task$formulas),identical(record$adduct,task$adduct))
  list(path=task$path,adduct=task$adduct,batch=task$batch,formulas=length(task$formulas),reused=reused)
}

ufa_merge_parts <- function(parts) {
  fields <- c("MassMAIso","MolecularFormula","IsotopicProfile","R13C","IndexMAIso","IPsize")
  db <- setNames(lapply(fields,function(field) {
    values <- lapply(parts,function(x)x[[field]])
    if (field=="IsotopicProfile") unlist(values,recursive=FALSE) else unlist(values,use.names=FALSE)
  }),fields)
  db$AggregatedList <- getFromNamespace("aggregatedIPdbListGenerator","IDSL.UFA")(db$MassMAIso)
  db[c("AggregatedList",fields)]
}

ufa_cached_database <- function(formulas,adduct,directory,merge_only=FALSE) {
  plan <- ufa_cache_plan(formulas,adduct,directory)
  record <- ufa_read_record(plan$path,plan$key,plan$spec)
  hit <- !is.null(record);built <- reused <- 0L
  if (!hit) record <- ufa_with_lock(plan$path,{
    parts <- lapply(plan$chunks,function(task) {
      chunk <- ufa_read_record(task$path,task$key,task$spec)
      if (is.null(chunk)) {
        if (merge_only) stop("Incomplete native profile universe: a committed batch is missing")
        receipt <- ufa_prepare_chunk(task)
        built <<- built+!receipt$reused;reused <<- reused+receipt$reused
        chunk <- ufa_read_record(task$path,task$key,task$spec)
      } else reused <<- reused+1L
      chunk$db
    })
    db <- ufa_merge_parts(parts)
    value <- list(key=plan$key,spec=plan$spec,adduct=adduct,formulas=plan$formulas,db=db)
    ufa_atomic_record(value,plan$path);value
  })
  stopifnot(identical(record$formulas,plan$formulas),identical(record$adduct,adduct))
  list(db=record$db,path=normalizePath(plan$path),cache_hit=hit,key=plan$key,
       sha256=digest::digest(file=plan$path,algo="sha256"),built_chunks=built,reused_chunks=reused)
}

ufa_subset_database <- function(database, ion_formulas) {
  stopifnot(!anyDuplicated(ion_formulas))
  index <- match(ion_formulas,database$MolecularFormula)
  if (anyNA(index)) stop("Native profile generation unavailable for a requested candidate")
  fields <- c("MassMAIso","MolecularFormula","IsotopicProfile","R13C","IndexMAIso","IPsize")
  result <- lapply(database[fields],function(x)x[index])
  result$AggregatedList <- getFromNamespace("aggregatedIPdbListGenerator","IDSL.UFA")(result$MassMAIso)
  result[c("AggregatedList",fields)]
}

ufa_prune_spectra <- function(spectra, database, accuracy=.01) {
  # Keep the union of EVERY native candidate/isotope matching window. The native
  # matcher still applies its own exact tolerance and nearest-centroid rule.
  centers <- sort(unique(unlist(lapply(database$IsotopicProfile,function(x)x[,1]))))
  guard <- 8*.Machine$double.eps*pmax(1,abs(centers))
  starts <- centers-accuracy-guard;ends <- centers+accuracy+guard
  lower <- upper <- numeric(0)
  for (i in seq_along(starts)) {
    j <- length(lower)
    if (j && starts[i]<=upper[j]) upper[j] <- max(upper[j],ends[i]) else {
      lower <- c(lower,starts[i]);upper <- c(upper,ends[i])
    }
  }
  lapply(spectra,function(s) {
    bin <- findInterval(s[,1],lower)
    keep <- bin>0L
    keep[keep] <- s[keep,1]<=upper[bin[keep]]
    s[keep,,drop=FALSE] # Preserve original order, including nearest-mass ties.
  })
}

ufa_extract_signal <- function(spectra, target, ppm=4) {
  answer <- t(vapply(spectra,function(s) {
    candidates <- which(abs(s[,1]-target)<=target*ppm*1e-6)
    if (!length(candidates)) return(c(intensity=0,mz=NA_real_,centroids=0))
    chosen <- candidates[which.min(abs(s[candidates,1]-target))]
    c(intensity=s[chosen,2],mz=s[chosen,1],centroids=length(candidates))
  },c(intensity=0,mz=0,centroids=0)))
  answer
}

ufa_signal_quality <- function(signal, settings=UFA_ANCHOR_SETTINGS) {
  raw <- signal[,"intensity"]
  y <- pmax(0,raw-as.numeric(quantile(raw,.2)))
  apex <- which.max(y);height <- max(y)
  if (height<=0 || length(raw)<settings$min_points) return(list(usable=FALSE,y=y,apex=apex))
  above <- which(y>=settings$relative_height*height)
  usable <- length(above)>=settings$min_points && sum(above<apex)>=settings$edges &&
    sum(above>apex)>=settings$edges && !any(signal[above,"centroids"]>1)
  list(usable=usable,y=y,apex=apex)
}

ufa_observed_anchor <- function(spectra, mono_mz, charge, times=NULL,
                                settings=UFA_ANCHOR_SETTINGS) {
  if (!length(spectra)) stop("Insufficient observed scans")
  if (!is.null(times)) {
    delta <- diff(times)
    if (any(!is.finite(delta)) || any(delta<=0) || max(delta)>3*median(delta))
      stop("Acquisition gap or invalid scan ordering")
  }
  z <- abs(charge)
  signals <- lapply(0:(settings$max_order+1L),function(k)
    ufa_extract_signal(spectra,mono_mz+k*1.003354835336/z,settings$ppm))
  quality <- lapply(signals,ufa_signal_quality,settings=settings)
  if (!quality[[1]]$usable) stop("Original monoisotopic trace is missing, sparse or ambiguous")
  reference <- quality[[1]]$y
  heights <- vapply(signals,function(s)max(s[,"intensity"]),numeric(1))
  accepted <- rep(FALSE,settings$max_order+1L);accepted[1] <- TRUE
  for (i in 2:length(accepted)) {
    q <- quality[[i]]
    if (!q$usable || abs(q$apex-quality[[1]]$apex)>1L) next
    r <- cor(reference,q$y)
    overlap <- sum(pmin(reference/sum(reference),q$y/sum(q$y)))
    accepted[i] <- is.finite(r) && r>=settings$shape_r && overlap>=settings$overlap
  }
  eligible <- which(accepted)
  selected <- eligible[which.max(heights[eligible])]
  competing <- which(heights[seq_along(accepted)] >= (1-settings$tie_fraction)*heights[selected])
  if (any(!accepted[competing])) stop("Strong interfering or unverified isotope-anchor signal")
  if (sum(heights[eligible] >= (1-settings$tie_fraction)*heights[selected])>1L)
    stop("Most-abundant isotope anchor is ambiguous")
  raw <- signals[[selected]]
  apex <- which.max(raw[,"intensity"])
  list(order=selected-1L,mz=raw[apex,"mz"],height=raw[apex,"intensity"],
       r13c=100*sum(signals[[selected+1L]][,"intensity"])/sum(raw[,"intensity"]),
       monoisotope_mz=mono_mz,charge=charge,scan_count=length(spectra))
}

ufa_annotate <- function(db,spectra,anchor,rt,prune=TRUE) {
  peaklist <- matrix(0,nrow=1,ncol=11)
  peaklist[1,c(1,2,3,4,8,11)] <- c(1,length(spectra),rt,anchor$height,anchor$mz,anchor$r13c)
  used <- if (prune) ufa_prune_spectra(spectra,db) else spectra
  IDSL.UFA::molecular_formula_annotator(db,used,peaklist,1,massAccuracy=.01,maxNEME=5,
    minPCS=950,minNDCS=2,minRCS=5,scoreCoefficients=rep(1,5),number_processing_threads=1)
}

ufa_native_candidates <- function(result,query) {
  if (is.null(result)) return(list())
  ions <- vapply(query$candidates,function(c)c$ion_formula,character(1))
  index <- match(result$IonFormula,ions)
  counts <- table(ions)
  if (anyNA(index) || any(counts[result$IonFormula]!=1L))
    stop("Native ion composition has an ambiguous neutral-formula crosswalk")
  lapply(seq_len(nrow(result)),function(i)list(
    formula=query$candidates[[index[i]]]$formula,adduct=query$adduct,charge=query$charge,
    rank=as.numeric(result$rank[i]),native_ion_formula=result$IonFormula[i],
    PCS=as.numeric(result$PCS[i]),NEME=as.numeric(result$normEucMassError[i]),
    NDCS=as.integer(result$NDCS[i]),RCS=as.numeric(result$`ratioChromScan(%)`[i])))
}
