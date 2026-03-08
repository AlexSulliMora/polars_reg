# Benchmark R fixest for comparison with polars_reg
# Called by generate_chart.py via subprocess
#
# Usage: Rscript bench_r.R <csv_path> <reps> <outfile>
# Output: CSV file with benchmark,time_ms

suppressPackageStartupMessages({
  library(fixest)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
reps <- as.integer(args[2])
outfile <- args[3]

dt <- fread(csv_path)

bench <- function(name, expr) {
  # warmup
  eval(expr)
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()["elapsed"]
    eval(expr)
    times[i] <- (proc.time()["elapsed"] - t0) * 1000
  }
  med <- sort(times)[ceiling(reps / 2)]
  cat(sprintf("%s,%.3f\n", name, med), file = outfile, append = TRUE)
}

# Clear outfile
cat("", file = outfile)

bench("OLS", quote(feols(y ~ x1 + x2, data = dt)))
bench("OLS + robust SE", quote(
  feols(y ~ x1 + x2, data = dt, vcov = "hetero")
))
bench("OLS + clustered SE", quote(
  feols(y ~ x1 + x2, data = dt, cluster = ~firm_id)
))
bench("1-way FE + cluster", quote(
  feols(y ~ x1 + x2 | firm_id, data = dt, cluster = ~firm_id)
))
bench("2-way FE + cluster", quote(
  feols(y ~ x1 + x2 | firm_id + year_id, data = dt, cluster = ~firm_id)
))
bench("2SLS / IV", quote(
  feols(y ~ x1 | x_endog ~ z1 + z2, data = dt)
))
bench("High-dim FE", quote(
  feols(
    y ~ x1 + x2 | firm_id + industry_id, data = dt,
    cluster = ~ firm_id + industry_id
  )
))
bench("PPML (Poisson)", quote(
  fepois(y_count ~ x1 + x2, data = dt)
))
