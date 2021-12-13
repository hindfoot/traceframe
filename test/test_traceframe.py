import unittest
import pandas as pd
import tracesframe as tf
import datetime

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

    def emptyTraces(self):
        return pd.DataFrame({"traceID": [],
                "traceName": [],
                "nspans": [],
                "errspans": [],
                "duration": [],
                "startTime": [],
                "processes": [],
                })

    def emptyScans(self):
        return pd.DataFrame({"traceID": [],
                "traceName": [],
                "nspans": [],
                "errspans": [],
                "duration": [],
                "startTime": [],
                "service": [],
                "parent": [],
                })

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

