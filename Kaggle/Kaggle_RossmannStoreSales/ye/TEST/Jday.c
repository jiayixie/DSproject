#include <iostream>
#include <cstdlib>

int jday ( int y, int m, int d ) {
   int i, jd = 0;
   for( i = 1; i < m; i++ ) {
      if ( (i==1) || (i==3) || (i==5) || (i==7) || (i==8) || (i==10) ) jd += 31;
      else if (i==2) {
         if ( (y%400==0) || (y%100!=0&&y%4==0) ) jd += 29;
         else jd += 28;
      }
      else jd += 30;
   }
   return jd + d;
}

int main( int argc, char* argv[] ) {
   if( argc != 4 ) {
      std::cerr<<"Usage: "<<argv[0]<<" [year] [month] [day]"<<std::endl;
      return -1;
   }
   int y=atoi(argv[1]);
   int m=atoi(argv[2]);
   int d=atoi(argv[3]);
   std::cout << jday(y,m,d) << std::endl;
   return 0;
}

