from tweepy.streaming import StreamListener
from tweepy import Stream
from tweepy import OAuthHandler
from colorama import Fore,Back,Style,init
init()
import csv,json
import time
import sys
import pandas as pd
# class for Streaming tweets
#StdOutListener class
class StdOutListener(StreamListener):
    #Intialised function when mining start
    tweet_no=0
    def __init__(self, max_tweets,api=None):
        self.max_tweets = max_tweets
        self.api=api
        #create a file with data& current time
        self.filename = 'data '+'_'+'trump'+'.csv'
       #self.filename1 = 'data'+'_'+time.strftime('%y%m%d-%H%M%S')+'.json'
        #create a new file with filename
        csvFile = open(self.filename,'w',encoding='utf-8')
        #jsonFile = open(self.filename1,'w')
        #create a csvWriter
        csvWriter = csv.writer(csvFile)
        #write a single row with the headers of the columns
        csvWriter.writerow(['created_at',
                    'user.location',
                    'user.followers_count',
                    'user.friends_count',
                    'text'])

    #when tweets appear
    def on_status(self, status):
        #jp=open('final.json','a')
        """if not 'RT @' in status.text:
            try:
                #print(status)
                j1=json.dumps(status._json)
                j=json.loads(j1)
                #jw=json.dump(j,jp)
                print(j['id'])
                df = pd.DataFrame(j)
                print(df.to_csv(sep='\t', index=False))
            except Exception as e:
                # print error
                print(e)
                pass
        twe = json.load(status)"""

        #open csv file previously created
        csvFile = open(self.filename,'a',encoding='utf-8')
        #jsonFile = open(self.filename1,'a')
        #create csv Writer
        csvWriter = csv.writer(csvFile)
        #tweet is not a retweet
        if not 'RT @' in status.text:
            self.tweet_no += 1
            try:
                print (Style.BRIGHT,Fore.CYAN+'tweet number: {}'.format(self.tweet_no), status.text, status.user.location)
                #write tweet info to file created
                print(Style.RESET_ALL)
                csvWriter.writerow([status.created_at,
                                    status.user.location,
                                    status.user.followers_count,
                                    status.user.friends_count,
                                    status.text])
                #jsonWriter =json.dumps(status+'\n',jsonFile)
            except Exception as e:
                #print error
                print(e)
                pass
            if self.tweet_no >= self.max_tweets:
                sys.exit('Limit of ' + str(self.max_tweets) + ' tweets reached.')
        csvFile.close()
        return

    #error
    def on_error(self, status_code):
        #print error code
        print(Back.RED+"error status_code:",status_code)
        print(Style.RESET_ALL)
        #if error is 401, status is bad credentials
        if status_code == 401:
            #end the stream
            return False
        if status_code == 420:
            print(Back.RED+"Rate limited")
            print(Style.RESET_ALL)
            return False
    #delete tweet appears
    def on_delete(self, status_id, user_id):
        #print message
        print(Back.RED+"Delete notice")
        print(Style.RESET_ALL)
        return
    #when time_out
    def on_timeout(self):
        #print message
        print(sys.stderr,'timeout...')
        #wait 10 seconds
        time.sleep(10)
        return
    


#function for mining
def start_mining(q,count):
    consumer_key = 'ee740gS08Rn7MiDneuRsS9YhZ'
    consumer_secret_key = '7fe9zos5F8uNvhqMLpwp98q57v4oHR6reOAYdO5Xi6n9Qq1qA6'
    access_token = '987024507608141824-qxZd4VK4ou1zajN4hpTRRZ62UCWJIWG'
    access_token_secret = 'om3YJyfuipcdLDRtlEkhpCb41djy3MHM5Sa8N2MYinpxj'
    #create listener
    l = StdOutListener(count)
    #create auth info
    auth = OAuthHandler(consumer_key,consumer_secret_key)
    auth.set_access_token(access_token,access_token_secret)
    #stream object
    stream = Stream(auth,l)

    stream.filter(track=q,languages=["en"], is_async=True)
    
if __name__ == '__main__':
    print(Style.BRIGHT,Fore.GREEN+'enter HashTag or Keyword to Search:')
    print(Style.RESET_ALL)
    q1=[input()]
    print('\n')
    print(Style.BRIGHT,Fore.BLUE+"enter count or limit to Download Tweets:")
    print(Style.RESET_ALL)
    count1= int(input())
    start_mining(q1,count1)
    
