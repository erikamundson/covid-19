#!/usr/bin/env python
# coding: utf-8
import pandas as pd

dates = ["Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28","Mar 29","Mar 30","Mar 31","Apr 01","Apr 02","Apr 03","Apr 04","Apr 05","Apr 06","Apr 07","Apr 08","Apr 09","Apr 10","Apr 11","Apr 12","Apr 13","Apr 14","Apr 15","Apr 16","Apr 17","Apr 18","Apr 19","Apr 20","Apr 21","Apr 22","Apr 23","Apr 24","Apr 25","Apr 26","Apr 27","Apr 28","Apr 29","Apr 30","May 01","May 02","May 03","May 04","May 05","May 06","May 07","May 08","May 09","May 10","May 11","May 12","May 13","May 14","May 15","May 16","May 17","May 18","May 19","May 20","May 21","May 22","May 23","May 24","May 25","May 26","May 27","May 28","May 29","May 30","May 31","Jun 01","Jun 02","Jun 03","Jun 04","Jun 05","Jun 06","Jun 07","Jun 08","Jun 09","Jun 10","Jun 11","Jun 12","Jun 13","Jun 14","Jun 15","Jun 16","Jun 17","Jun 18","Jun 19","Jun 20","Jun 21","Jun 22","Jun 23","Jun 24","Jun 25","Jun 26","Jun 27","Jun 28","Jun 29","Jun 30", "Jul 01","Jul 02","Jul 03","Jul 04","Jul 05","Jul 06","Jul 07","Jul 08","Jul 09","Jun 10","Jul 11","Jul 12","Jul 13","Jul 14","Jul 15","Jul 16","Jul 17","Jul 18","Jul 19","Jul 20","Jul 21"]

total_cases = [75,100,124,158,221,319,435,541,704,994,1301,1631,2185,2774,3622,4611,6366,9333,13935,19608,24498,33946,44325,55579,69136,86668,105584,125250,145526,168822,194114,221225,251796,284759,319444,345345,377095,411166,443680,477766,512010,542498,570358,597452,624893,655569,685712,718208,747401,773598,801820,828058,858363,890436,929559,965082,991670,1014937,1040510,1069105,1100105,1136266,1166135,1193573,1218332,1243241,1268819,1298462,1327754,1353989,1374807,1393513,1416537,1438927,1467022,1494646,1518978,1539179,1562722,1583523,1606195,1635039,1659705,1681644,1701574,1721494,1740894,1761648,1784733,1810430,1834193,1854948,1877379,1899666,1920497,1943347,1968774,1991636,2010569,2029638,2048723,2069763,2093253,2120658,2146148,2166329,2187230,2212864,2239121,2267085,2300667,2334098,2360216,2391754,2427820,2466254,2506490,2553855,2597454,2638017,2682781,2728856,2781217,2838739,2898358,2948357,2994393,3045164,3101020,3163318,3224892,3297170,3359174,3417795,3483584,3549632,3621637,3695025,3770012,3833271,3898550,3961429,4028569]

new_cases = [7,25,24,34,63,98,116,106,163,290,307,330,554,589,848,989,1755,2967,4602,5673,4890,9448,10379,11254,13557,17532,18916,19666,20276,23296,25292,27111,30571,32963,34685,25901,31750,34071,32514,34086,34244,30488,27860,27094,27441,30676,30143,32496,29193,26197,28222,26238,30305,32073,39123,35523,26588,23267,25573,28595,31000,36161,29869,27438,24759,24909,25578,29643,29292,26235,20818,18706,23024,22390,28095,27624,24332,20201,23543,20801,22672,28844,24666,21939,19930,19920,19400,20754,23085,25697,23763,20755,22431,22287,20831,22850,25427,22862,18933,19069,19085,21040,23490,27405,25490,20181,20901,25634,26257,27964,33582,33431,26118,31538,36066,38434,40236,47365,43599,40563,44764,46075,52361,57522,59619,49999,46036,50771,55856,62298,61574,72278,62004,58621,65789,66048,72005,73388,74987,63259,65279,62879,67140]

deaths_total = [1,6,9,11,12,15,19,22,26,30,38,41,48,58,73,95,121,171,240,310,376,512,693,962,1269,1628,2129,2779,3280,4102,5201,6460,7671,8953,10530,11953,13476,15743,17953,20099,22379,24445,26201,27947,30573,33263,35516,38111,40031,41625,43605,46353,48762,51163,53154,55263,56433,57835,60371,62812,65085,67031,68761,69936,71274,73677,76251,78423,80136,81587,82802,83876,85788,87645,89435,91071,92309,93191,94209,95790,97222,98652,99974,101027,101653,102289,103072,104635,105873,107106,108139,108790,109485,110632,111736,112786,113773,114490,114874,115472,116576,117574,118490,119290,120004,120340,120772,121630,122449,123205,123934,124516,124786,125155,126026,126845,127498,128161,128673,128958,129324,130050,130726,131413,132039,132305,132568,132946,133939,134830,135791,136639,137371,137752,138217,139153,140155,141118,142064,142877,143289,143834,144953]

new_deaths = [0,5,3,2,1,3,4,3,4,4,8,3,7,10,15,22,26,50,69,70,66,136,181,269,307,359,501,650,501,822,1099,1259,1211,1282,1577,1423,1523,2267,2210,2146,2280,2066,1756,1746,2626,2690,2253,2595,1920,1594,1980,2748,2409,2401,1991,2109,1170,1402,2536,2441,2273,1946,1730,1175,1338,2403,2574,2172,1713,1451,1215,1074,1912,1857,1790,1636,1238,882,1018,1581,1432,1430,1322,1053,626,636,783,1563,1238,1233,1033,651,695,1147,1104,1050,987,717,384,598,1104,998,916,800,714,336,432,858,819,756,729,582,270,369,871,819,653,663,512,285,366,726,676,687,626,266,263,378,993,891,961,848,732,381,465,936,1002,963,946,813,412,545,1119]



#Create a dictionary then turn that into a dataframe
data = {}
for i in range(len(dates)):
    data[i] = [dates[i], new_cases[i], total_cases[i], new_deaths[i], deaths_total[i]]
df = pd.DataFrame.from_dict(data, orient='index')
df = df.rename(columns={0: 'Date', 1: 'New Cases', 2: 'Total Cases', 3: 'New Deaths', 4: 'Total Deaths'})