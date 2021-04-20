//Project UID db1f506d06d84ab787baf250c265e24e
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <math.h>
#include "csvstream.h"

using namespace std;
using _pair_=std::pair<string, int>;

typedef pair<string, double> PAIR;

bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {
  return lhs.second < rhs.second;
}

struct CmpByValue {
  bool operator()(const PAIR& lhs, const PAIR& rhs) {
    return lhs.second < rhs.second;
  }
};



class classifier_trainer{
    public:

    classifier_trainer(){}

    ~classifier_trainer(){};

    int get_post_num() const{return posts;}

    void add_word_to_bag(const set<string> &post){
        set<string>::iterator it = post.begin();
        for(;it!=post.end();++it){       
            word_bag.insert(*it);
        }
    }

    int num_uniq_words() const{
        return word_bag.size();
    }

    void add_post(const set<string>& input){
        set<string>::iterator it = input.begin();
        pair<string,int> word_post;
        for(;it!=input.end();++it){
            if(word_count.find(*it)==word_count.end()){
                word_post = {*it,1};
                word_count.insert(word_post);
            }else{
                ++word_count[*it];
            }
        }
        ++posts;
    }
    
    void debug(){
        cout << "vocabulary size = "<< word_bag.size()<<endl<<endl;
        cout << "classes:"<<endl;
        map<string,int>::const_iterator it = label_count.begin();
        for(;it!=label_count.end();++it){
            cout <<"  "<<it->first<<", "<<it->second
            <<" examples, log-prior = "<< log(double(it->second)/double(posts)) << endl;
        }
        cout << "classifier parameters:"<<endl;
        auto it1 = word_label_count.begin();
        for(;it1!=word_label_count.end();++it1){
            pair<string,string> p = it1->first;
            cout << "  " <<it1->first.first <<":"<<it1->first.second<<
            ", count = "<<it1->second<<", log-likelihood = "<< w_given_c(p)<<endl;
        }
    }


    void count_label(const string &label){
        if(label_count.find(label)==label_count.end()){
            _pair_ new_label = {label, 1};
            label_count.insert(new_label);
        }else{
            ++label_count[label];
        }
    }

    double w_given_c(pair<string,string> &p){
        double num_C_with_W = word_label_count[p];
        double num_C = label_count[p.first];
        return log(num_C_with_W/num_C);
    }

    pair<string,double> log_prob_score(const set<string> & input){
        double result = 0;
        map<string,double> test_prob;

        map<string, int>::iterator  label = label_count.begin();
        for(;label!=label_count.end();++label){//prob for each label
            result += log(double(label->second)/double(posts));//log-prior of label C

            set<string>::iterator word = input.begin();
            for(;word!=input.end();++word){
                pair<string,string> p = {label->first,*word};
                if(word_label_count.find(p)!=word_label_count.end()){
                    //if w was seen with label c
                
                    result += w_given_c(p);
                }
                else if(word_bag.find(*word)!=word_bag.end()){
                    //if not seen with label C but seen in training data
                    
                    result += log(double(word_count[*word])/double(posts));
                }else{//never seen
                
                    result += log(1/double(posts));

                }
            }
            std::pair<string, double> prob = {label->first, result};
            test_prob.insert(prob);//store value
            result = 0;//reset result
        }

        return *max_element(test_prob.begin(),test_prob.end(),CmpByValue());
    }

    void add_word_label(const string& label, const set<string>& words){
        pair<string,string> label_word;
        set<string>::const_iterator it = words.begin();
        for(;it!=words.end();++it){
            label_word = {label,*it};
            if(word_label_count.find(label_word)==word_label_count.end()){
                pair<pair<string,string>,int> new_pair = {label_word,1};
                word_label_count.insert(new_pair);
            }else{
                ++word_label_count[label_word];
            }
        }
    }

    private:

    int posts = 0;
    map<string, int> word_count;
    set<string> word_bag;
    map<string, int> label_count;
    map<pair<string,string>,int> word_label_count;




};

static set<string> unique_words(const string &str) {
        istringstream source{str};
        return {istream_iterator<string>{source},
         istream_iterator<string>{}};
    }




int main(int argc, char** argv){
    cout.precision(3);
    if(argc!=3&&argc!=4){
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return -1;
    }
    ifstream train,test;
    train.open(argv[1]);
    test.open(argv[2]);
    if(!train.is_open()){
        cout << "Error opening file: " << argv[1] << endl;
    }
    if(!test.is_open()){
        cout << "Error opening file: " << argv[2] << endl;
    }
    csvstream csvin(train);
    classifier_trainer classifier;

    vector<pair<string, string>> row;

    if(argc == 4){
        cout << "training data:"<<endl;
    }

    while (csvin >> row) {
        string label;
        for (auto &col:row) {
            const string &column_name = col.first;
            const string &datum = col.second;
            if(column_name == "tag"){
                classifier.count_label(datum);
                label = datum;
                if(argc==4){
                    cout << "  label = " <<datum << ", ";
                }
            }
            if(column_name == "content"){
                set<string> input = unique_words(datum);
                classifier.add_post(input);
                classifier.add_word_to_bag(input);
                classifier.add_word_label(label,input);
                if(argc==4){
                    cout << "content = " <<datum <<endl;
                }
            }
        }
    }


    cout << "trained on "<< classifier.get_post_num() << " examples"<<endl;
    if(argc==4){
        classifier.debug();
    }
    cout << endl;
    csvstream csv_test(test);
    cout <<"test data:"<<endl;
    int rows = 0;
    int correct = 0;
    string true_label;
    while(csv_test>>row){
        ++rows;
        for (auto &col:row) {
            const string &column_name = col.first;
            const string &datum = col.second;
            if(column_name == "tag"){
                cout << "  correct = " << datum <<", ";
                true_label = datum;
            }
            if(column_name == "content"){
                set<string> input1 = unique_words(datum);
                auto result = classifier.log_prob_score(input1);
                string label_pred = result.first;
                double prob = result.second;
                cout << "predicted = " << label_pred 
                <<", log-probability score = " << prob<<endl;
                cout << "  content = " << datum << endl<<endl;
                input1.clear();
                if(label_pred == true_label)++correct;
            }
        }
    }
    cout << "performance: "<< correct 
    << " / " << rows<<" posts predicted correctly" << endl;
    return 0;

}