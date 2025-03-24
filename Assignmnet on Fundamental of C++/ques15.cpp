#include <iostream>
using namespace std;
int main(){
    int a;
    cin>>a;
 int mp=1;
    while(a>0){
    mp=mp+a*a;
    a--;
    
    }
   cout<<"Answer"<<mp;
}