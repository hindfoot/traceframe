import requests
import time
import datetime
from collections import defaultdict

import pandas as pd
import holoviews as hv
hv.extension('bokeh')
from bokeh.models.formatters import DatetimeTickFormatter

# TODO Support passwords or token/certs if Jaeger is deployed secured
def known_services(http_endpoint):
    # TODO Switch to logging
    print(f"Querying Jaeger for known services")

    # 45s timeout too slow when all-in-one loaded with 15 minutes of 100_000_spans_per_second.json
    try:
        start_time = time.time()
        resp=requests.get(f"{http_endpoint}/api/services", timeout=45)
        if time.time() - start_time > 1:
            print(f"Jaeger /api/services took {time.time() - start_time} seconds")
    except requests.exceptions.Timeout:
        print(f"Querying Jaeger for services timed out, aborting")
        assert(False)

    if resp.status_code != 200: print(f"/api/services status_code {resp.status_code}")
    services=resp.json()["data"]
    print(f"Jaeger reports {len(services)} service(s)")
    # if "jaeger-query" in services: services.remove("jaeger-query")
    return services

def get_traces(jaeger_http_endpoint, jaeger_password, service, operation, 
        tagexpr, start, end, mindur, maxdur, limit):
    print("in get_traces")
    if jaeger_password is not None:
        raise "Jaeger password UNIMPLEMENTED"
    if operation is not None:
        raise "operation param UNIMPLEMENTED"
    if tagexpr is not None:
        raise "tagexpr param UNIMPLEMENTED"
    if start is not None:
        raise "start param UNIMPLEMENTED"
    if end is not None:
        raise "start param UNIMPLEMENTED"
    if mindur is not None:
        raise "mindur param UNIMPLEMENTED"
    if maxdur is not None:
        raise "mindur param UNIMPLEMENTED"
    if limit is None:
        raise "requests without limit UNIMPLEMENTED"
    if limit > 1500:
        raise "limit>1500 UNIMPLEMENTED"

    if service is None:
        raise Exception("unspecified service name UNIMPLEMENTED")

    resp = requests.get(f"{jaeger_http_endpoint}/api/traces",
        params={'service':service, 'limit': limit}, timeout=30)
    if resp.status_code != 200:
        raise f"/api/traces resp.status_code={resp.status_code}"

    # print(resp.json())
    traces=resp.json()["data"]
    return traces

# Given a Python object (or pandas row), return the root span as a dict
def rootspan(row):
    for span in row.spans:
       if len(span["references"]) == 0:
            return span
    return None

# Return a name for this trace for a human user
def traceobj_name(row):
    root = rootspan(row)
    if root is None:
        # Fallback if there are no root spans
        return row.traceID

    # return f'{row["processes"][root["processID"]]["serviceName"]}: {root["operationName"]}'
    # rootID = root["processID"]
    # proc = row["processes"][rootID]
    # svcName = proc["serviceName"]
    return "{}: {}".format(row["processes"][root["processID"]]["serviceName"], root["operationName"])


def traceobj_spancount(row):
    return len(row.spans)

def traceobj_errcount(row):
    count = 0
    for span in row.spans:
        for tag in span["tags"]:
            if tag["key"] == "error":
                count = count + 1
    return count

# Return a Python Datetime
def traceobj_starttime(row):
    root = rootspan(row)
    if root is None:
        # Fallback if there are no root spans
        return datetime.datetime.fromtimestamp(0)
    
    return datetime.datetime.fromtimestamp(root["startTime"]/1000000.0)

# Return a Python Timedelta
def traceobj_duration(row):
    root = rootspan(row)
    if root is None:
        # Fallback if there are no root spans
        return datetime.timedelta(0)

    # TODO If the root span ends while another span is doing async work, we should measure against the last span's completion
    return datetime.timedelta(microseconds=root["duration"])

def process_traces(dfRawTraces):
    # print("dfRawTraces=", dfRawTraces)
    if dfRawTraces.empty:
        dfTraces = pd.DataFrame({"traceID": [],
                "traceName": [],
                "nspans": [],
                "errspans": [],
                "duration": [],
                "startTime": [],
                "processes": [],
                })
    else:
        dfTraces = pd.DataFrame({"traceID": dfRawTraces["traceID"],
                "traceName": dfRawTraces.apply(traceobj_name, axis=1),
                "nspans": dfRawTraces.apply(traceobj_spancount, axis=1),
                "errspans": dfRawTraces.apply(traceobj_errcount, axis=1),
                "duration": dfRawTraces.apply(traceobj_duration, axis=1),
                "startTime": dfRawTraces.apply(traceobj_starttime, axis=1),
                "processes": dfRawTraces.apply(lambda row: row["processes"], axis=1),
                })
    dfTraces["iserror"] = dfTraces["errspans"] > 0
    return dfTraces.sort_values(by='startTime', ascending=False)


def from_jaeger(jaeger_http_endpoint, jaeger_password=None, service=None, operation=None, 
        tagexpr=None, start=None, end=None, mindur=None, maxdur=None, limit=None):
    print("in from_jaeger")
    traces = get_traces(jaeger_http_endpoint, jaeger_password, service,
        operation, tagexpr, start, end, mindur, maxdur, limit)
    print(f"back from get_traces, got {len(traces)} traces")
    dfRawTraces = pd.DataFrame(traces)
    return process_traces(dfRawTraces)


def from_es(es_endpoint, es_password, prefix, service, start, end):
    print("in from_es")
    raise "UNIMPLEMENTED"
    return None

def pretty_duration(dur):
   return f"{int(dur.microseconds/1000)}ms"

def color_nonzero_red(val):
    # return 'color: {}'.format('red' if val != 0 else 'black')
    return 'color: red; background-color: pink' if val != 0 else ''


def flag_nonzero(val):
    if val == 0:
        return val
    return f"<span color='red'>{val}</span>"

def pretty_trace_table(jaeger_http_endpoint, dfPage):
    return (dfPage.style
         .hide_columns(["iserror", "processes"])
         .format({"duration": pretty_duration})
         .applymap(color_nonzero_red, subset=['errspans'])
         .format(lambda val: f'<a href="{jaeger_http_endpoint}/trace/{val}">{val}</a>', subset=['traceID'])
        )

def parent_span(span):
  for reference in span["references"]:
     if reference["refType"] == "CHILD_OF":
        return reference["spanID"]
  return ""

def spans_from_jaeger(jaeger_http_endpoint, jaeger_password=None, services=[], operation=None, 
        tagexpr=None, start=None, end=None, mindur=None, maxdur=None, limit=None):
    print("in spans_from_jaeger")
    if not services:
        raise Exception("at least one service name required")

    # Fetch traces from all services
    svc_traces=dict()
    for service in services:
        traces = get_traces(jaeger_http_endpoint, jaeger_password, service,
            operation, tagexpr, start, end, mindur, maxdur, limit)
        print(f"back from get_traces(..., {service}), got {len(traces)} traces")
        svc_traces[service]=traces

    spans_with_service=[]
    for service in services:
        for trace in svc_traces[service]:
            for span in trace["spans"]:
                copy = dict(span)
                # copy["service"] = service
                copy["service"] = trace["processes"][span["processID"]]["serviceName"]
                copy["parent"] = parent_span(span)
                del copy["references"]
                for tag in span["tags"]:
                    copy[tag["key"]] = tag["value"]
                del copy["tags"]
                spans_with_service.append(copy)

    dfSpans = pd.DataFrame(spans_with_service)
    return dfSpans

class Span:
    def __init__(self, traceID, spanID, operationName, startTime, duration, parent_span_id):
        if startTime < 16740000000:
            raise Exception("Invalid timestamp (pass nanos)")
        # if startTime > 16740000000:
        #    raise Exception("Invalid timestamp (don't pass nanos)")
        self.traceID = traceID
        self.spanID = spanID
        self.operationName = operationName
        self.startTime = startTime
        self.duration = duration
        self.parent_span_id = parent_span_id

class SpanEvent:
    def __init__(self, isCall, time_stamp, span):
        if time_stamp < 16740000000:
            raise Exception("Invalid timestamp (pass nanos)")
        # if time_stamp > 16740000000:
        #    raise Exception("Invalid timestamp (don't pass nanos)")

        self.isCall = isCall
        self.time_stamp = time_stamp
        self.span = span
        # print(f"created {self}")

    def __repr__(self):
        return "{} {} {}/{}".format(
            "Call" if self.isCall else "Return",
            datetime.fromtimestamp(self.time_stamp/1000000),
            self.span['spanID'],
            self.span['label']
        )

class CritSeg:
    def __init__(self, startTime, duration, span):
        if startTime < 16740000000:
            raise Exception("Invalid timestamp (pass nanos)")
        # if startTime > 16740000000:
        #     raise Exception("Invalid timestamp (don't pass nanos)")
        # assert isinstance(span, Span)
        self.startTime = startTime
        self.duration = duration
        self.span = span
        # print(f"created critseg {self}")

    def __repr__(self):
        return "start={} duration={} end={} spanID={}".format(
            datetime.fromtimestamp(self.startTime/1000000), # .strftime("%H:%M:%S"),
            self.duration,
            datetime.fromtimestamp((self.startTime+self.duration)/1000000),
            self.span['spanID']
        )

# Given tree of spans, calculate critical path segments
# (For explanation, see _Distributed Tracing in Practice_ by Parker, page 160)
def get_critical_segments(spans):
    if len(spans) == 0:
        raise Exception("No spans")

    id_to_span={"": None}
    events=[]
    for span in spans:
        id_to_span[span["spanID"]] = span
        events.append(SpanEvent(True, span["startTime"], span))
        events.append(SpanEvent(False, span["startTime"]+span["duration"], span))
    events.sort(key=lambda event: event.time_stamp)
    # print("events=", events)

    crit_segs = []
    outstanding = defaultdict(lambda: []) # map[spanID][]spanIDs, all the active child spans
    leader_stack = []
    leader = events[0].span
    ts = leader['startTime']
    for event in events:
        # print(f"at {datetime.fromtimestamp(ts/1000000)} considering event {event}")
        parentSpanID = event.span["parent"]
        if event.isCall:
            if leader["spanID"] == parentSpanID and len(outstanding[parentSpanID]) == 0:
                if event.span["parent"] != "":
                    # print(f"waiting on no one, call event has parent, creating critical segment at {datetime.fromtimestamp(ts/1000000)} for parent {id_to_span[event.span['parent']]['label']}")
                    crit_segs.append(CritSeg(ts, event.time_stamp-ts, id_to_span[event.span["parent"]]))
                    leader_stack.append(leader)
                    leader = event.span
                ts = event.time_stamp
            
            outstanding[parentSpanID].append(event.span["spanID"])
        else: # not isCall
            # print(f"Returning from {event.span['spanID']}")
            # print(f"parentSpanID is {parentSpanID}, which has {len(outstanding[parentSpanID])} oustanding spans.  The spans are {outstanding[parentSpanID]}")
            outstanding[parentSpanID].remove(event.span["spanID"])

            if len(outstanding[leader["spanID"]]) == 0:
               # print(f"waiting on ourselves, return event, creating critical segment at {datetime.fromtimestamp(ts/1000000)} for span")
               crit_seg = CritSeg(ts, event.time_stamp-ts, leader)
               # print(f"waiting on ourselves, return event, creating critical segment {crit_seg}")
               crit_segs.append(crit_seg)
               ts = event.time_stamp
               # print(f"Returning, event.span is {event.span['label']}, with {len(outstanding[event.span['spanID']])} outstanding")
               while len(outstanding[leader["spanID"]]) == 0 and len(leader_stack) != 0:
                  leader = leader_stack.pop()

               if len(outstanding[leader["spanID"]]) != 0:
                   leader_stack.append(leader)
                   leader = id_to_span[outstanding[leader["spanID"]][0]]

               # print(f'The leader is now {leader["spanID"]}')
            # else:
            #     print(f"(The leader {leader['spanID']} has more outstanding spans, so this return does not change the leader)")
            
    return crit_segs

def traceWithSpans(dfT, dfS, traceID):
    # return {"traceID": traceID}
    rows = dfT[dfT["traceID"] == traceID]
    # display(rows)
    if rows.size < 1:
        raise Exception("Trace ID not found")

    retval = rows.to_dict(orient='records')[0]
    # spans = dfS[dfS["traceID"] == traceID]
    # Drop duplicates because perhaps the span table includes stuff from all queries (TODO: Remove in tracesframe library)
    spans = dfS[dfS["traceID"] == traceID].drop_duplicates(subset='spanID')
    retval["spans"] = spans.to_dict(orient='records')
    return retval

# traceWithSpans(dfT, dfSpans, 'efd0841560171ae4')

def showSingleTrace(trace):

    origspans = trace["spans"]
    if len(origspans) == 0:
        raise Exception("Trace has no spans")
    # services = trace["services"]

    id_to_span={'': {'label':''}}
    spans=[]
    for span in origspans:
        copy = dict(span)
        # TODO Consider adding 'label' to tracesframe library
        copy["label"] = f"{copy['service']}: {copy['operationName']}"
        copy["endTime"] = copy["startTime"] + copy["duration"]
        spans.append(copy)
        id_to_span[copy["spanID"]] = copy

    # spans.sort(key=lambda span: span["startTime"])
    # display(spans)

    dfSpans = pd.DataFrame(spans)
    # display(dfSpans)
    
    events = dfSpans["label"].unique()

    event = dfSpans["label"]
    # display(event)
    data = dict(
        start=dfSpans["startTime"],
        end=dfSpans["endTime"],
        service=dfSpans["service"],
        start_event = event,
        end_event = event
    )
    # display(data)

    # TODO Sort the Segments (sorting the data doesn't help)
    seg = hv.Segments(data, kdims=[hv.Dimension('start'), 
                             hv.Dimension('start_event', label='Span'), 'end', 'end_event'],
                     vdims=["service"])
    spanSegs = seg.opts(title=f"Critical path for trace {trace['traceID']}",
             height=len(events)*20+80, 
             width=1000, 
             line_width=10, 
             color="service", 
             xformatter=DatetimeTickFormatter(milliseconds=["%d %b %Y"]),
            show_legend=False)
    
    crits = get_critical_segments(spans)
    crit_data = dict(
        start = [crit.startTime for crit in crits],
        end = [crit.startTime + crit.duration for crit in crits],
        service = [crit.span["service"] for crit in crits],
        start_event = [crit.span["label"] for crit in crits],
        end_event = [crit.span["label"] for crit in crits],
    )
    # display(crit_data)
    crit_seg = hv.Segments(crit_data, kdims=[hv.Dimension('start'), 
                             hv.Dimension('start_event', label='Span'), 'end', 'end_event'],
                     vdims=["service"])
    spanCrits = crit_seg.opts(height=len(events)*20+80, width=1000, line_width=6, color="red", xformatter=DatetimeTickFormatter(milliseconds=["%d %b %Y"]))
    # display(spanCrits)

    dfSpans["parentLabel"] = dfSpans["parent"].apply(lambda parentid: id_to_span[parentid]["label"])
    spansWithParents = dfSpans[dfSpans["parent"] != ""]
    callData = dict(
        start=spansWithParents["startTime"],
        end=spansWithParents["startTime"],
        start_event = spansWithParents["parentLabel"],
        end_event = spansWithParents["label"]
    )
    returnData = dict(
        start= spansWithParents["endTime"],
        end= spansWithParents["endTime"],
        start_event =  spansWithParents["label"],
        end_event = spansWithParents["parentLabel"]
    )
    callTree = hv.Segments(callData, kdims=['start', hv.Dimension('start_event', label='Span'), 'end', 'end_event']).opts(color='black')
    returnTree = hv.Segments(returnData, kdims=['start', hv.Dimension('start_event', label='Span'), 'end', 'end_event']).opts(color='black')
    # return callTree * returnTree * seg
    return spanSegs * spanCrits * callTree * returnTree
