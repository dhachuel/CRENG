import numpy as np
import networkx as nx
import datetime
import random

##
## GLOBAL PARAMETERS
##
DECAY_TIME_UNIT = 'years'
LOGISTIC_MAP_ALPHA_COEF = 1
PAGE_RANK_ALPHA = 0.9


##
## SIMULATION PARAMETERS
##
RND_INIT_DATE = datetime.datetime.strptime('1/1/2008 1:30 PM', '%m/%d/%Y %I:%M %p')
RND_END_DATE = datetime.datetime.strptime('1/1/2017 4:50 AM', '%m/%d/%Y %I:%M %p')
TOPICS = ['blockchain', 'machine learning', 'data science', 'climate', 'healthcare', 'popcorn','cornholing']
N_USERS = 10
SAMPLES = 200


##
## GLOBAL HELPERS
##
def logisticMap(x:float, alpha: (int, float)) -> float:
	return(1/(1 + np.e**(-alpha*x)))

def getScore(user_id: int, topics: list, scores: dict) -> float:
	return(
		np.mean([scores[topic][user_id] for topic in topics])
	)

##
## SIMULATION HELPERS
##
def randomDate(start: datetime.datetime, end: datetime.datetime) -> datetime.datetime:
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return(start + datetime.timedelta(seconds=random_second))

def calculateEndorsementTimeDecay(created: datetime.datetime, decay_time_unit) -> float:
	now = datetime.datetime.now()
	delta = now - created
	days = delta.days
	time_units = {
		'weeks' : 7,
		'months' : 30,
		'hours' : (1/24),
		'minutes' : (1/1440),
		'seconds' : (1/86399),
		'years' : 365,
		'days' : 1
	}
	return(1/(days/time_units[decay_time_unit]))

def randomEndorsement(n_nodes: int) -> (list, np.array):
	node_pair = np.random.choice(n_nodes, 2, replace=False)

	# USER_0, USER_1, TOPIC, CREATED
	return ([
		node_pair[0],
		node_pair[1],
		TOPICS[random.randint(0, len(TOPICS) - 1)],
		randomDate(start=RND_INIT_DATE, end=RND_END_DATE)
	])

def generateRandomEndorsements(n_users: int, n_samples: int) -> (list, np.array):
	return (
		np.array(
			[randomEndorsement(N_USERS) for i in range(SAMPLES)]

		)
	)







##
## TESTING
##
raw_data = generateRandomEndorsements(n_users=N_USERS, n_samples=SAMPLES)

adj_mats = {}
for topic in TOPICS:
	adj_mats[topic] = np.zeros((N_USERS, N_USERS))

for row in raw_data:
	adj_mats[row[2]][row[0], row[1]] += calculateEndorsementTimeDecay(created=row[3], decay_time_unit=DECAY_TIME_UNIT)

page_ranks = {}
for topic in TOPICS:
	G = nx.DiGraph(adj_mats[topic])
	topic_page_rank = nx.pagerank(G, alpha=PAGE_RANK_ALPHA)
	topic_page_rank_mean = np.mean(list(topic_page_rank.values()))
	topic_page_rank_sd = np.std(list(topic_page_rank.values()))

	page_ranks[topic] = {}
	for user in topic_page_rank.keys():
		page_ranks[topic][user] = logisticMap(
			x=((topic_page_rank[user]/topic_page_rank_mean)-1),
			alpha=LOGISTIC_MAP_ALPHA_COEF
		)


getScore(
	user_id=3,
	topics=['climate','popcorn'],
	scores=page_ranks
)