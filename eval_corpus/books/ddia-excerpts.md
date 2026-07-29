# Designing Data-Intensive Applications — Excerpts

## Chapter 5: Replication

Replication means keeping a copy of the same data on multiple machines that are connected via a network. Reasons for replication include keeping data geographically close to users, allowing the system to continue working even if some of its parts have failed, and scaling out the number of machines that can serve read queries.

### Leaders and Followers

The most common approach to replication is called leader-based replication (also known as active/passive or master-slave replication). One replica is designated the leader (also known as master or primary). When clients want to write to the database, they must send their requests to the leader, which first writes the new data to its local storage. The other replicas are called followers (read replicas, slaves, secondaries, or hot standbys). Whenever the leader writes new data to its local storage, it also sends the data change to all followers as part of a replication log or change stream. Each follower takes the log from the leader and updates its local copy of the database accordingly, by applying all writes in the same order as they were processed on the leader. When a client wants to read from the database, it can query either the leader or any follower. However, writes are only accepted on the leader.

### Synchronous vs Asynchronous Replication

An important trade-off in replicated systems is whether the replication is synchronous or asynchronous. In synchronous replication, the leader waits until the follower has confirmed that it received the write before reporting success to the user, and before making the write visible to other clients. In asynchronous replication, the leader sends the message but doesn't wait for a response from the follower. The advantage of synchronous replication is that the follower is guaranteed to have an up-to-date copy of the data that is consistent with the leader. The disadvantage is that if the synchronous follower doesn't respond, the write cannot be processed at all.

### Setting Up New Followers

Setting up a follower can often be done without downtime. Theoretically, you could simply copy the leader's data files to the follower and start replicating from that point. However, copying the files requires a filesystem snapshot consistent with the leader's replication log position.

## Chapter 6: Partitioning

Partitioning is the splitting of a large dataset into smaller subsets. Each partition is a small database of its own. Reasons for partitioning include scalability and query throughput. Different partitions can be placed on different nodes in a shared-nothing cluster. A database query can potentially be parallelized across many nodes.

### Partitioning of Key-Value Data

One way of partitioning is to assign each key to a partition by taking the hash of the key and then assigning a range of hashes to each partition. This approach is known as hash-based partitioning or consistent hashing. It has the advantage of distributing keys quite evenly across partitions.

The opposite approach is range-based partitioning, where keys are assigned to partitions based on their actual value ranges. Range-based partitioning is more effective for range-scan queries and when the data has a natural sort order.

### Partitioning and Secondary Indexes

Secondary indexes are the main factor that makes partitioning complicated. The problem is that secondary indexes don't map neatly to partitions. There are two main approaches to partitioning a database with secondary indexes: document-based partitioning (where each partition maintains its own secondary indexes, also known as local indexes) and term-based partitioning (where a global index is partitioned by the indexed term rather than the document).

## Chapter 7: Transactions

The concept of a transaction has been the mechanism for grouping multiple reads and writes into a logical unit. Transactions simplify the programming model by hiding the complexities of concurrency and partial failures.

### ACID

ACID stands for Atomicity, Consistency, Isolation, and Durability. Atomicity means that writes in a transaction are executed as a whole — either all succeed or all fail. Consistency means that the database is in a valid state before and after the transaction. Isolation means that concurrently executing transactions are isolated from each other. Durability means that once a transaction has committed successfully, the data will not be forgotten, even if there is a hardware fault or the database crashes.

### Isolation Levels

Read Committed is the most basic level of isolation. It guarantees that any data read is committed at the moment it is read — it never sees uncommitted data. Snapshot Isolation (also known as Repeatable Read) gives each transaction a consistent snapshot of the database at a point in time. Serializable isolation is the strongest level; it guarantees that even though transactions may execute concurrently, the end result is the same as if they had executed one at a time, in some serial order.

## Chapter 10: Batch Processing

Batch processing is the classic approach to processing large volumes of data. A batch processing job takes a fixed input and produces a fixed output, with no side effects or external interactions during the processing.

### MapReduce

MapReduce is a programming model for processing large datasets in a distributed fashion. A MapReduce job has two phases: the mapper and the reducer. The mapper applies a transformation to each input record independently. The reducer aggregates the mapper's outputs. Between the map and reduce phases, the framework sorts and groups the intermediate data by key.

## Chapter 11: Stream Processing

Stream processing deals with data that arrives continuously, as opposed to batch processing where the input is bounded. The key difference is that streams are unbounded — you never know when the data will end.

### Event Streams

An event is a small, self-contained record of something that happened at some point in time. Events are written by producers and read by consumers. A stream is the sequence of events ordered by time.

### Representing Event Streams

A log is the most natural representation of an event stream. In the context of stream processing, a log is simply an append-only sequence of records. A log can be partitioned across multiple machines, similar to a partitioned database table.
