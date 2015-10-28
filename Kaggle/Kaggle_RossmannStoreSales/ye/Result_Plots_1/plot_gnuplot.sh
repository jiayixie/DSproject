#!/bin/bash

fcoef=coef_linearR.txt
fdata=train_DailyRecs_s0001.txt
fout=$fdata".ps"
ftemp=.formula.tmp
awk '{printf $1; for(i=2;i<NF;i++){printf "+$%d*(%f) ",i+1,$(i)}}END{print "\n"}' $fcoef > $ftemp

gnuplot <<- END
	set term postscript enhanced color
	set out '$fout'
	set multiplot layout 3,1
	set xrange[0:365]
	plot '$fdata' u 18:2 w l, '$fdata' u 18:(`more $ftemp`) w l title 'prediction (year 1)'
	set xrange[365:730]
	plot '$fdata' u 18:2 w l, '$fdata' u 18:(`more $ftemp`) w l title 'prediction (year 2)'
	set xrange[730:1095]
	plot '$fdata' u 18:2 w l, '$fdata' u 18:(`more $ftemp`) w l title 'prediction (year 3)'
	q
END
rm -f $ftemp
echo $fout
