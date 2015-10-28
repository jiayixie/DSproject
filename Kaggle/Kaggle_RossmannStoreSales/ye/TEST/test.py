import datetime

date1 = datetime.date(2012,10,13)
date2 = datetime.date(2012,11,1)
delta = date1-date2
print date1
print date2
print delta
print delta.total_seconds()
print delta.total_seconds()<0

yearweek = "2015 0"
print datetime.datetime.strptime(yearweek + ' 1 1', "%Y %W %w %M")
print datetime.datetime.strptime(yearweek + ' 2 1', "%Y %W %w %M")
print datetime.datetime.strptime('2015 1 2 1', "%Y %W %w %M") - datetime.timedelta(days=7)

print datetime.datetime.strptime('2015 Jan 01', '%Y %b %d')
print datetime.datetime.strptime('2015 deC 01', "%Y %b %d").date()


