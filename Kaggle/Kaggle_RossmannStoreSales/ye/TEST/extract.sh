#!/bin/bash

fin=../Kaggle/Kaggle_RossmannStoreSales/train.csv
exeJday=./Jday
storeN=1
fout=data_store${storeN}.txt; rm -f $fout

Nday2013=`$exeJday 2013 12 31`
Nday2014=`$exeJday 2014 12 31`
awk -F, -v storeN=$storeN '$1==storeN&&$5>0{print $3,$4,$5,$2}' $fin | while read date sales Ncustomers ndweek; do
   date=`echo $date | awk -F- '{print $1,$2,$3}'`
   y=`echo $date | awk '{print $1}'`
   m=`echo $date | awk '{print $2}'`
   d=`echo $date | awk '{print $3}'`
   Nday=`$exeJday $date`
   if [ $y -gt 2013 ]; then
      Nday=`echo $Nday $Nday2013 | awk '{print $1+$2}'`
   fi
   if [ $y -gt 2014 ]; then
      Nday=`echo $Nday $Nday2014 | awk '{print $1+$2}'`
   fi
   echo $sales $Nday $Ncustomers $y $m $d $ndweek >> $fout
done


