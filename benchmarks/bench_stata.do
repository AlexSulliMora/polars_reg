* Benchmark Stata for comparison with polars_reg
* Called by generate_chart.py via subprocess
*
* Usage: stata -b do bench_stata.do csv_path reps outfile
* Reads data from CSV, runs benchmarks, writes results to outfile
* Results written after each benchmark so partial results survive timeouts

local csv_path "`1'"
local reps = `2'
local outfile "`3'"

quietly {
    import delimited "`csv_path'", clear
}

* Install reghdfe if needed
capture which reghdfe
if _rc {
    quietly ssc install reghdfe, replace
    quietly ssc install ftools, replace
}

* Clear output file
tempname fh
file open `fh' using "`outfile'", write replace
file close `fh'

* --- OLS ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reg y x1 x2
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "OLS,`med'" _n
file close `fh'

* --- OLS + robust ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reg y x1 x2, robust
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "OLS + robust SE,`med'" _n
file close `fh'

* --- OLS + cluster ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reg y x1 x2, vce(cluster firm_id)
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "OLS + clustered SE,`med'" _n
file close `fh'

* --- 1-way FE + cluster (reghdfe) ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reghdfe y x1 x2, absorb(firm_id) vce(cluster firm_id)
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "1-way FE + cluster,`med'" _n
file close `fh'

* --- 2-way FE + cluster (reghdfe) ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reghdfe y x1 x2, absorb(firm_id year_id) vce(cluster firm_id)
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "2-way FE + cluster,`med'" _n
file close `fh'

* --- 2SLS ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly ivregress 2sls y x1 (x_endog = z1 z2)
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "2SLS / IV,`med'" _n
file close `fh'

* --- High-dim FE + 2-way cluster ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly reghdfe y x1 x2, absorb(firm_id industry_id) vce(cluster firm_id industry_id)
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "High-dim FE,`med'" _n
file close `fh'

* --- PPML (Poisson) ---
local total = 0
forvalues i = 1/`reps' {
    timer clear 1
    timer on 1
    quietly poisson y_count x1 x2
    timer off 1
    quietly timer list 1
    local total = `total' + r(t1)
}
local med = (`total' / `reps') * 1000
file open `fh' using "`outfile'", write append
file write `fh' "PPML (Poisson),`med'" _n
file close `fh'

exit, clear
