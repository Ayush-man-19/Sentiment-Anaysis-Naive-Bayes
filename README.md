# Sentiment-Anaysis-Naive-Bayes
Sentiment Analysis of tweets using the Naive Bayes Classifier

In this project I used the Naive Bayes classifier algorithm that calculates the probability of occurence of an event based on the probability of another given event.
Let E1, E2,…, En be a set of events associated with a sample space S, where all the events E1, E2,…, En have nonzero probability of occurrence and they form a partition of S. Let A be any event associated with S, then according to Bayes theorem,

P(Ei|A) = P(Ei).P(A|Ei)/[ ∑P(Ek).P(A|Ek)]

Summation from k=1 to k = n

Naive baiyes is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature

For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.

#Approach
This section is devoted to binary sentiment analysis using the Naive Bayes classifier with multinomial distribution. We go through the brief overview of constructing a classifier from the probability model, then move to data preprocessing, training and hyperparameters optimization stages. 

Given tweet: “Today is a good day!”

We process the tweet and remove articles, prepositions etc.
Next we find the key words in the tweet

Key words = [“today”, “good”, “day”])

We calculate the probability of each key word being in a positive and negative tweet

P(tweet being positive | [“today”, “good”, “day”])
P(tweet being negative | [“today”, “good”, “day”])

A frequency dictionary tracks the number of times a specific word is found in a positive or negative tweet.
We employ the frequency dictionary in order to calculate the probability of the word being in a positive or negative tweet
We have three tweets which are pre-defined as being negative or positive.

	“1” is assigned to a positive tweet
	“0” is assigned to a negative tweet

[“today”, “good”, “day”] = 1
[“watch”, “bad”, “movie”] = 0
[“movie”, “scary”, “but”, “bad”] = 0

#Steps Followed
1. Importing the required libraries and modules
2. Downloading the twitter dataset
3. Pre-processing data to remove insignificant words and characters
4. Remove hyperlinks, Twitter marks and styles
5. Tokenize the string
6. Remove stop words and punctuations
7. Stemming
8. Splitting the data into training and testing
9. Create a frequency dictionary
10. Train model using Naive Bayes and predict



