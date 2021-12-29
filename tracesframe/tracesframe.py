from bokeh.models.formatters import DatetimeTickFormatter
import requests
import time
import datetime
import json
from collections import defaultdict
from typing import List, Dict, TextIO, Optional, Union, Any

import elasticsearch

import pandas as pd
import holoviews as hv
hv.extension('bokeh')

# Some Jaeger backends have API has a limit
# We can't ask for more than 1500 traces at a time with Badger backend
JAEGER_MAX_TRACES_RETURNABLE = 1500

# By default Jaeger cleans up after 7 days.  TODO Let people change this.``
MAX_LOOKBACK_IN_DAYS = 14
MAX_LOOKBACK_IN_SECONDS = MAX_LOOKBACK_IN_DAYS * 24 * 60 * 60

# TODO Support passwords or token/certs if Jaeger is deployed secured


def known_services(http_endpoint: str) -> List[str]:
    # TODO Switch to logging
    print(f"Querying Jaeger for known services")

    # 45s timeout too slow when all-in-one loaded with 15 minutes of 100_000_spans_per_second.json
    try:
        start_time = time.time()
        resp = requests.get(f"{http_endpoint}/api/services", timeout=45)
        if time.time() - start_time > 1:
            print(
                f"Jaeger /api/services took {time.time() - start_time} seconds")
    except requests.exceptions.Timeout:
        print(f"Querying Jaeger for services timed out, aborting")
        assert(False)

    if resp.status_code != 200:
        print(f"/api/services status_code {resp.status_code}")
    services = resp.json()["data"]
    print(f"Jaeger reports {len(services)} service(s)")
    # if "jaeger-query" in services: services.remove("jaeger-query")
    return services

# Uses INTERNAL (unofficial API) https://www.jaegertracing.io/docs/1.25/apis/#http-json-internal


def get_traces(jaeger_http_endpoint: str, jaeger_password: Optional[str], service: Optional[str],
               operation: Optional[str], tagexpr: Optional[str],
               start: Optional[int], end: Optional[int], mindur: Optional[int], maxdur: Optional[int],
               limit: Optional[int]):
    print(f"in get_traces, start={start} end={end}")
    if jaeger_password is not None:
        raise Exception("Jaeger password UNIMPLEMENTED")
    if operation is not None:
        raise Exception("operation param UNIMPLEMENTED")
    if tagexpr is not None:
        raise Exception("tagexpr param UNIMPLEMENTED")
    if start is not None and type(start) != int:
        raise Exception("start param, if supplied, must be int")
    if end is not None and type(end) != int:
        raise Exception("end param, if supplied, must be int")
    if mindur is not None:
        raise Exception("mindur param UNIMPLEMENTED")
    if maxdur is not None:
        raise Exception("mindur param UNIMPLEMENTED")

    if service is None:
        raise Exception("unspecified service name UNIMPLEMENTED")

    # If we don't specify a limit on the Jaeger query we'll only get 100 traces.
    locallimit = limit if limit is not None else 1500

    params: Dict[str, Union[int, str]] = {'service': service, 'limit': locallimit}
    if start is not None:
        params['start'] = str(start)
    if end is not None:
        params['end'] = str(end)
    # Note that we are talking to the V2 HTTP API.  It would be better to use gRPC or the V3 HTTP API.
    resp = requests.get(f"{jaeger_http_endpoint}/api/traces",
                        params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"/api/traces resp.status_code={resp.status_code}")

    traces = resp.json()["data"]
    if len(traces) < JAEGER_MAX_TRACES_RETURNABLE:
        print(
            f"Returning {len(traces)} traces directly from Jaeger; start={start} end={end}")
        return traces

    # We got the clipped / max number of traces.  That means traces were lost.
    # (Or that there were exactly JAEGER_MAX_TRACES_RETURNABLE matches)

    # Throw away results.  Ask in batches for smaller time windows
    # NOTE: A better strategy might be to keep the traces, and set `end`
    #   to the earliest trace-1.
    # NOTE: Hunting for the data in a library that talks to the Jaeger API is inefficient.
    #   A better approach might be to query the storage layer directly, or let Jaeger do that.

    # The query might not have start/end times; if not, use [maxlookback .. now]
    end = end if end else int(time.time()*1000)*1000
    start = start if start else int(
        time.time()-MAX_LOOKBACK_IN_SECONDS)*1000*1000
    print(
        f"Discarding {len(traces)} traces; preparing to recurse in get_traces, start={start} end={end}")

    midpoint = int((start + end) / 2)
    # SINGLY OR DOUBLY RECURSIVE
    print(f"Doing additional query for second half of time range")
    second_batch = get_traces(jaeger_http_endpoint, jaeger_password, service, operation,
                              tagexpr, midpoint+1, end, mindur, maxdur, limit)

    if limit is not None and len(second_batch) >= limit:
        print(f"Second half query returning {len(second_batch)} traces")
        # TODO Sort by trace startTime, so we throw away the earliest
        return second_batch[-limit::]

    print(f"Doing additional query for first half of time range")
    remaining_limit = limit - len(second_batch) if limit is not None else None
    first_batch = get_traces(jaeger_http_endpoint, jaeger_password, service, operation,
                             tagexpr, start, midpoint, mindur, maxdur, remaining_limit)

    traces = first_batch + second_batch
    if limit is not None:
        print(f"Clipping {len(traces)} to {limit}")
        # TODO Sort by trace startTime, so we throw away the earliest
        return traces[-limit::]

    return traces


# Given a Python object (or pandas row), return the root span as a dict
def rootspan(row) -> Optional[Dict]:
    # print(f"REACHED rootspan for row {row}")
    for span in row.spans:
        if len(span["references"]) == 0:
            return span

    # We did not find the root span.  That means this trace
    # is missing the root span.
    # return None
    # TODO Log Warn?  Info?
    if len(row.spans) == 0:
        return None
    # TODO Instead of picking at random, pick the span with the earliest startTime
    # min_time = min(row.spans, key=lambda span: span["startTime"])
    # return next(filter(lambda span: span['startTime'] == min_time, row.spans), None)
    return row.spans[0]


# Return a name for this trace for a human user
def traceobj_name(row) -> str:
    root = rootspan(row)
    if root is None:
        # Fallback if there are no root spans
        print(f"Trace {row['traceID']} has no root span")
        return row.traceID

    # return f'{row["processes"][root["processID"]]["serviceName"]}: {root["operationName"]}'
    # rootID = root["processID"]
    # proc = row["processes"][rootID]
    # svcName = proc["serviceName"]
    return "{}: {}".format(row["processes"][root["processID"]]["serviceName"], root["operationName"])


def traceobj_spancount(row) -> int:
    return len(row.spans)


def traceobj_errcount(row) -> int:
    count = 0
    for span in row.spans:
        for tag in span["tags"]:
            if tag["key"] == "error":
                count = count + 1
    return count

# Return a Python Datetime


def traceobj_starttime(row) -> datetime.datetime:
    root = rootspan(row)
    if root is None:
        # Fallback if there are no root spans
        return datetime.datetime.fromtimestamp(0)

    return datetime.datetime.fromtimestamp(root["startTime"]/1000000.0)

# Return a Python Timedelta


def traceobj_duration(row) -> datetime.timedelta:
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


def traces_from_jaeger(jaeger_http_endpoint: str, jaeger_password: Optional[str] = None,
                       service: Optional[str] = None, operation: Optional[str] = None,
                       tagexpr: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None,
                       mindur: Optional[int] = None, maxdur: Optional[int] = None, limit: Optional[int] = None):
    return from_jaeger(jaeger_http_endpoint, jaeger_password, service, operation,
                       tagexpr, start, end, mindur, maxdur, limit)

# TODO Deprecate


def from_jaeger(jaeger_http_endpoint: str, jaeger_password: str = None, service: str = None,
                operation: str = None,
                tagexpr: str = None, start=None, end=None, mindur: int = None, maxdur: int = None, limit: int = None):
    print("in from_jaeger")
    traces = get_traces(jaeger_http_endpoint, jaeger_password, service,
                        operation, tagexpr, start, end, mindur, maxdur, limit)
    print(f"back from get_traces, got {len(traces)} traces")
    dfRawTraces = pd.DataFrame(traces)
    return process_traces(dfRawTraces)


def traces_from_jaeger_file(f: TextIO):
    traces = json.load(f)
    dfRawTraces = pd.DataFrame(traces)
    return process_traces(dfRawTraces)


def taglist_to_tags(taglist: List[Dict[str, Any]]) -> Dict[str, Any]:
    retval = {}
    for kv in taglist:
        retval[kv['key']] = kv['value']
    return retval


def traces_from_es(es_endpoint: str, es_password: str, prefix: str,
                   service: Optional[str] = None, operation: Optional[str] = None, tagexpr: Optional[str] = None,
                   start: Optional[int] = None, end: Optional[int] = None, mindur: Optional[int] = None, maxdur: Optional[int] = None, limit: Optional[int] = None):

    # TODO Figure out what to do about 'limit'...
    raw_spans = internal_spans_from_es(es_endpoint, es_password, prefix, service, operation,
                                       tagexpr, start, end, mindur, maxdur, limit)
    print(f"back from internal_spans_from_es, got {len(raw_spans)} spans")

    raw_spans.sort(key=lambda span: span['traceID'])

    # spans have keys ['traceID', 'spanID', 'flags', 'operationName', 'references', 'startTime', 'startTimeMillis', 'duration', 'tags', 'logs', 'process']
    PROCESS_NAME_KEY = 'hostname'
    all_traces = []
    if len(raw_spans) > 0:
        trace_spans: List[Dict[str, Any]] = []
        trace_processes: Dict[str, str] = {}
        traceID = raw_spans[0]["traceID"]
        for span in raw_spans:
            if span["traceID"] != traceID:
                all_traces.append({
                    'traceID': traceID,
                    'spans': trace_spans,
                    # 'nspans': len(trace_spans),
                    'processes': trace_processes,
                })
                trace_spans = []
                trace_processes = {}
                traceID = span["traceID"]

            tags = taglist_to_tags(span['process']['tags'])
            span['processID'] = tags[PROCESS_NAME_KEY]
            trace_spans.append(span)
            trace_processes[tags[PROCESS_NAME_KEY]] = span['process']
            # print(f"SpanID {span['spanID']} has process {span['process']}")

        all_traces.append({
            'traceID': traceID,
            'spans': trace_spans,
            # 'nspans': len(trace_spans),
            'processes': trace_processes,
        })

    # debugID = '4501fbafa0a79cff'
    # debugTrace = next(filter(lambda trace: trace['traceID'] == debugID, all_traces), None)
    # print(debugTrace)
    # dfRawTraces = pd.DataFrame([debugTrace])

    dfRawTraces = pd.DataFrame(all_traces)
    return process_traces(dfRawTraces)


def spans_from_es(es_endpoint: str, es_password: str, prefix: str,
                  service: Optional[str] = None, operation: Optional[str] = None,
                  tagexpr: Optional[str] = None,
                  start: Optional[int] = None, end: Optional[int] = None, mindur: Optional[int] = None, maxdur: Optional[int] = None,
                  limit: Optional[int] = None):
    spans = internal_spans_from_es(es_endpoint, es_password, prefix, service, operation,
                                   tagexpr, start, end, mindur, maxdur, limit)
    raise Exception("UNIMPLEMENTED")


# Note: experimental.  Do not use.
def internal_spans_from_es(es_endpoint: str, es_password: str, prefix: str,
                           service: Optional[str] = None, operation: Optional[str] = None,
                           tagexpr: Optional[str] = None,
                           start: Optional[int] = None, end: Optional[int] = None, mindur: Optional[int] = None, maxdur: Optional[int] = None,
                           limit: Optional[int] = None):
    print("in internal_spans_from_es")

    if operation is not None:
        raise Exception("operation param UNIMPLEMENTED")
    if tagexpr is not None:
        raise Exception("tagexpr param UNIMPLEMENTED")
    if start is not None:
        raise Exception("start param UNIMPLEMENTED")
    if end is not None:
        raise Exception("end param UNIMPLEMENTED")
    if mindur is not None:
        raise Exception("mindur param UNIMPLEMENTED")
    if maxdur is not None:
        raise Exception("maxdur param UNIMPLEMENTED")

    # TODO Pick better default
    if limit is None:
        limit = 2000

    # TODO Support verify != False
    # TODO Support other ES usernames besides 'elastic'
    # resp=requests.post(f"{es_endpoint}/_search",
    #               headers={'Content-Type': 'application/json'},
    #               verify=False,
    #               auth=requests.auth.HTTPBasicAuth('elastic', es_password),
    #               data='{"from" : 0, "size" : 9999, "query": { "wildcard": { "_index": "my-prefix-jaeger-span*" }}}',
    #               timeout=45)

    # TODO Support verify_certs != False
    # TODO Support other ES usernames besides 'elastic'
    client = elasticsearch.Elasticsearch(es_endpoint,
                                         http_auth=('elastic', es_password),
                                         verify_certs=False,
                                         scheme="https",
                                         port=443)  # TODO Why 443?

    search_body = {
        "size": min(10000, limit),
        "query": {
            "match_all": {}
        }
    }
    # TODO Don't use -*, use -2021-08-06 or whatever for start/stop
    data = client.search(
        index=f"{prefix}jaeger-span-*",
        body=search_body,
        scroll='15s'
    )

    all_spans: List[Dict[str, Any]] = []
    # position = 0
    scroll_size = len(data['hits']['hits'])
    scroll_id = data['_scroll_id']
    while scroll_size > 0:
        all_spans = all_spans + \
            list(map(lambda hit: hit["_source"], data["hits"]["hits"]))
        if len(all_spans) >= limit:
            break

        data = client.scroll(scroll_id=scroll_id, scroll='15s')

        # Update
        scroll_id = data['_scroll_id']
        scroll_size = len(data['hits']['hits'])
        # position = position + scroll_size

    return all_spans


def pretty_duration(dur) -> str:
    return f"{int(dur.microseconds/1000)}ms"


def color_nonzero_red(val) -> str:
    # return 'color: {}'.format('red' if val != 0 else 'black')
    return 'color: red; background-color: pink' if val != 0 else ''


def flag_nonzero(val) -> str:
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


def parent_span(span: Dict[str, Any]) -> str:
    for reference in span["references"]:
        if reference["refType"] == "CHILD_OF":
            return reference["spanID"]
    return ""


def spans_from_jaeger(jaeger_http_endpoint: str, jaeger_password: Optional[str] = None,
                      services: List[str] = [], operation: Optional[str] = None,
                      tagexpr: Optional[str] = None,
                      start: Optional[int] = None, end: Optional[int] = None, mindur: Optional[int] = None, maxdur: Optional[int] = None,
                      limit: Optional[int] = None):
    print("in spans_from_jaeger")
    if not services:
        raise Exception("at least one service name required")

    # Fetch traces from all services
    svc_traces = dict()
    for service in services:
        traces = get_traces(jaeger_http_endpoint, jaeger_password, service,
                            operation, tagexpr, start, end, mindur, maxdur, limit)
        print(
            f"back from get_traces(..., {service}), got {len(traces)} traces")
        svc_traces[service] = traces

    spans_with_service = []
    for service in services:
        # TODO refactor to use append_spans()
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


def spans_from_jaeger_file(f: TextIO):
    traces = json.load(f)

    spans_with_service: List[Dict[str, Any]] = []
    for trace in traces:
        append_spans(spans_with_service, trace)

    dfSpans = pd.DataFrame(spans_with_service)
    return dfSpans


def append_spans(spans: List[Dict[str, Any]], trace: Dict[str, Any]) -> None:
    for span in trace["spans"]:
        copy = dict(span)
        # copy["service"] = service
        copy["service"] = trace["processes"][span["processID"]]["serviceName"]
        copy["parent"] = parent_span(span)
        del copy["references"]
        for tag in span["tags"]:
            copy[tag["key"]] = tag["value"]
        del copy["tags"]
        spans.append(copy)


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
            datetime.datetime.fromtimestamp(
                self.startTime/1000000),  # .strftime("%H:%M:%S"),
            self.duration,
            datetime.datetime.fromtimestamp(
                (self.startTime+self.duration)/1000000),
            self.span['spanID']
        )

# Given tree of spans, calculate critical path segments
# (For explanation, see _Distributed Tracing in Practice_ by Parker, page 160)


def get_critical_segments(spans: List[Dict[str, Any]]):
    if len(spans) == 0:
        raise Exception("No spans")

    id_to_span: Dict[str, Optional[Dict[str, Any]]] = {"": None}
    events = []
    for span in spans:
        # span must be a dict
        id_to_span[span["spanID"]] = span
        events.append(SpanEvent(True, span["startTime"], span))
        events.append(
            SpanEvent(False, span["startTime"]+span["duration"], span))
    events.sort(key=lambda event: event.time_stamp)
    # print("events=", events)

    crit_segs = []
    # map[spanID][]spanIDs, all the active child spans
    outstanding: Dict[str, List[str]] = defaultdict(lambda: [])
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
                    crit_segs.append(
                        CritSeg(ts, event.time_stamp-ts, id_to_span[event.span["parent"]]))
                    leader_stack.append(leader)
                    leader = event.span
                ts = event.time_stamp

            outstanding[parentSpanID].append(event.span["spanID"])
        else:  # not isCall
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


def traceWithSpans(dfT, dfS, traceID: str) -> Dict:
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

# Show the spans on a timeline, similar to Jaeger, but with the critical path highlighted


def showSingleTrace(trace):

    origspans = trace["spans"]
    if len(origspans) == 0:
        raise Exception("Trace has no spans")
    # services = trace["services"]

    id_to_span = {'': {'label': ''}}
    spans = []
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
        start_event=event,
        end_event=event
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
                        xformatter=DatetimeTickFormatter(
                            milliseconds=["%d %b %Y"]),
                        show_legend=False)

    crits = get_critical_segments(spans)
    crit_data = dict(
        start=[crit.startTime for crit in crits],
        end=[crit.startTime + crit.duration for crit in crits],
        service=[crit.span["service"] for crit in crits],
        start_event=[crit.span["label"] for crit in crits],
        end_event=[crit.span["label"] for crit in crits],
    )
    # display(crit_data)
    crit_seg = hv.Segments(crit_data, kdims=[hv.Dimension('start'),
                                             hv.Dimension('start_event', label='Span'), 'end', 'end_event'],
                           vdims=["service"])
    spanCrits = crit_seg.opts(height=len(events)*20+80, width=1000, line_width=6,
                              color="red", xformatter=DatetimeTickFormatter(milliseconds=["%d %b %Y"]))
    # display(spanCrits)

    dfSpans["parentLabel"] = dfSpans["parent"].apply(
        lambda parentid: id_to_span[parentid]["label"])
    spansWithParents = dfSpans[dfSpans["parent"] != ""]
    callData = dict(
        start=spansWithParents["startTime"],
        end=spansWithParents["startTime"],
        start_event=spansWithParents["parentLabel"],
        end_event=spansWithParents["label"]
    )
    returnData = dict(
        start=spansWithParents["endTime"],
        end=spansWithParents["endTime"],
        start_event=spansWithParents["label"],
        end_event=spansWithParents["parentLabel"]
    )
    callTree = hv.Segments(callData, kdims=['start', hv.Dimension(
        'start_event', label='Span'), 'end', 'end_event']).opts(color='black')
    returnTree = hv.Segments(returnData, kdims=['start', hv.Dimension(
        'start_event', label='Span'), 'end', 'end_event']).opts(color='black')
    return spanSegs * spanCrits * callTree * returnTree
