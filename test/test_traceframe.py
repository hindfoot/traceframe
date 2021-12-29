import unittest
import pandas as pd
import traceframe as tf
import datetime
import json

# The critical path for test trace b6b80525a332cb6b
crit_path_b6b80525a332cb6b = '''
[
  {
    "startTime": 1639415450569000,
    "duration": 58000,
    "span": {
      "traceID": "b6b80525a332cb6b",
      "spanID": "b6b80525a332cb6b",
      "operationName": "/cart",
      "startTime": 1639415450569000,
      "duration": 291000,
      "processID": "p1",
      "parent": "",
      "service": "frontend"
    }
  },
  {
    "startTime": 1639415450627000,
    "duration": 131000,
    "span": {
      "traceID": "b6b80525a332cb6b",
      "spanID": "69dc17050f4c2221",
      "operationName": "/GetCart",
      "startTime": 1639415450627000,
      "duration": 162000,
      "processID": "p2",
      "parent": "b6b80525a332cb6b",
      "service": "cartservice"
    }
  },
  {
    "startTime": 1639415450758000,
    "duration": 31000,
    "span": {
      "traceID": "b6b80525a332cb6b",
      "spanID": "69dc17050f4c2221",
      "operationName": "/GetCart",
      "startTime": 1639415450627000,
      "duration": 162000,
      "processID": "p2",
      "parent": "b6b80525a332cb6b",
      "service": "cartservice"
    }
  },
  {
    "startTime": 1639415450789000,
    "duration": 11000,
    "span": {
      "traceID": "b6b80525a332cb6b",
      "spanID": "23890e1f876db993",
      "operationName": "/GetRecommendations",
      "startTime": 1639415450631000,
      "duration": 169000,
      "processID": "p3",
      "parent": "b6b80525a332cb6b",
      "service": "recommendationservice"
    }
  },
  {
    "startTime": 1639415450800000,
    "duration": 60000,
    "span": {
      "traceID": "b6b80525a332cb6b",
      "spanID": "b6b80525a332cb6b",
      "operationName": "/cart",
      "startTime": 1639415450569000,
      "duration": 291000,
      "processID": "p1",
      "parent": "",
      "service": "frontend"
    }
  }
]
'''

class TraceFrameTest(unittest.TestCase):

    def test(self):
        self.assertTrue(True)

    def testTraceWithSpans(self):
        dfT = self.trace_b6b80525a332cb6b()
        self.assertEqual(len(dfT.index), 1)

        dfSpans = self.spans_b6b80525a332cb6b()
        self.assertEqual(len(dfSpans.index), 4)

        leastSpansInATrace = dfT["nspans"].min()
        shortestTraceID = dfT[dfT["nspans"] == leastSpansInATrace].iloc[0].traceID
        trace = tf.traceWithSpans(dfT, dfSpans, shortestTraceID)

        self.assertEqual(len(trace["spans"]), len(dfSpans.index))

        # TODO Holoviews tests don't really fail when run outside Jupyter
        tf.showSingleTrace(trace)

    def testGetCriticalSegments(self):
        dfSpans = self.spans_b6b80525a332cb6b()
        spans = dfSpans.to_dict(orient='records')
        crit_segs = tf.get_critical_segments(spans)
        self.assertEqual(len(crit_segs), 5)
        self.assertEqual(json.dumps(crit_segs, default=vars, indent=2), crit_path_b6b80525a332cb6b.strip())


    def trace_b6b80525a332cb6b(self):
        return pd.DataFrame({"traceID": ['b6b80525a332cb6b'],
                "traceName": ['frontend: /cart'],
                "nspans": [4],
                "errspans": [0],
                "duration": [datetime.timedelta(microseconds=1639415450569000)],
                "startTime": [datetime.datetime.fromtimestamp(1639415450569000/1000000.0)],
                "processes": [self.exampleProcesses()],
                })

    def spans_b6b80525a332cb6b(self):
        return pd.DataFrame({
            "traceID": ['b6b80525a332cb6b', 'b6b80525a332cb6b', 'b6b80525a332cb6b', 'b6b80525a332cb6b'],
            "spanID": ['b6b80525a332cb6b', '69dc17050f4c2221', '23890e1f876db993', 'ee567b37267317a7'],
            "operationName": ['/cart', '/GetCart', '/GetRecommendations', '/GetProducts'],
            "startTime": [1639415450569000, 1639415450627000, 1639415450631000, 1639415450752000],
            "duration": [291000, 162000, 169000, 6000],
            "processID": ["p1", "p2", "p3", "p4"],
            "parent": ["", "b6b80525a332cb6b", "b6b80525a332cb6b", "23890e1f876db993"],
            "service": ["frontend", "cartservice", "recommendationservice", "productcatalogservice"],
            })

    def testJaegerResponse(self):

        with open("jaeger.json", "r") as f:
          dfT = tf.traces_from_jaeger_file(f)
          self.assertEqual(len(dfT), 100)
          self.assertEqual(dfT.size, 800)

        with open("jaeger.json", "r") as f:
          dfS = tf.spans_from_jaeger_file(f)
          self.assertEqual(len(dfS), 420)
          self.assertEqual(dfS.size, 29820)

    def testGetAllCriticalSegments(self):
        with open("jaeger.json", "r") as f:
          dfS = tf.spans_from_jaeger_file(f)

        traces = dict(tuple(dfS.groupby('traceID')))
        for traceID, dfTSpans in traces.items():
          # print(f"{traceID} has {len(dfTSpans)} spans")
          spans = dfTSpans.to_dict(orient='records')
          crit_segs = tf.get_critical_segments(spans)
          self.assertGreater(len(crit_segs), 0)

    def exampleProcesses(self):
        processes = {
            'p1': {
                'serviceName': 'frontend',
                'tags': [
                    {'key': 'hostname', 'type': 'string', 'value': 'Eds-64-Macbook.local'},
                    {'key': 'ip', 'type': 'string', 'value': '192.168.1.3'},
                    {'key': 'jaeger.version', 'type': 'string', 'value': 'Java-0.28.0'}
                    ]
                },
            'p2': {
                'serviceName': 'cartservice',
                'tags': [
                    {'key': 'hostname', 'type': 'string', 'value': 'Eds-64-Macbook.local'},
                    {'key': 'ip', 'type': 'string', 'value': '192.168.1.3'},
                    {'key': 'jaeger.version', 'type': 'string', 'value': 'Java-0.28.0'}
                ]
            },
            'p3': {
                'serviceName': 'recommendationservice',
                'tags': [
                    {'key': 'hostname', 'type': 'string', 'value': 'Eds-64-Macbook.local'},
                    {'key': 'ip', 'type': 'string', 'value': '192.168.1.3'},
                    {'key': 'jaeger.version', 'type': 'string', 'value': 'Java-0.28.0'}
                ]
            },
            'p4': {
                'serviceName': 'productcatalogservice',
                'tags': [
                    {'key': 'hostname', 'type': 'string', 'value': 'Eds-64-Macbook.local'},
                    {'key': 'ip', 'type': 'string', 'value': '192.168.1.3'},
                    {'key': 'jaeger.version', 'type': 'string', 'value': 'Java-0.28.0'}
                ]
            }
        }
        return processes

if __name__ == '__main__':
    unittest.main()

