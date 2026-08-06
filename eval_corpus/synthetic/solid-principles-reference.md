# SOLID Principles Reference Guide

## Introduction

SOLID is a mnemonic acronym for five design principles intended to make software designs more understandable, flexible, and maintainable. These principles apply to object-oriented design at the class and module level. When applied to a modern Python monorepo built on FastAPI and Pydantic v2, they provide a practical vocabulary for reviewing code: each principle names a specific kind of design smell and a concrete remedy.

The five principles are not rules to be applied dogmatically. They are heuristics that reduce the cost of change. A design that follows them tends to isolate the reasons a system changes, making each modification local, reviewable, and testable. The sections below give the definition of each principle, its application in a Python monorepo, and the common anti-patterns that violate it.

## Single Responsibility Principle

A class should have one, and only one, reason to change. This means each class should have a single, well-defined responsibility. When a class has multiple responsibilities, changes to one responsibility may affect the other, making the system fragile. The principle is about cohesion: the methods and attributes of a class should vary together, and a change to one requirement should not ripple through unrelated code. In a Python monorepo this means Pydantic models handle validation only, repositories handle data access only, services handle business logic only, and route handlers handle the HTTP contract only. Pydantic models are the boundary type for all I/O: they validate and serialize but do not talk to the database and do not contain business rules beyond structural validation. Repositories own all SQLAlchemy and query logic. Services orchestrate the business logic, calling repositories, applying transformations, and raising domain errors. Route handlers translate HTTP requests into service calls and translate results or exceptions back into HTTP responses.

A god class is a class that knows too much or does too much. For example, a UserManager class that validates data, creates database records, sends emails, and logs activity violates SRP. The symptom is a class with many unrelated methods and dependencies. A class like that becomes hard to test, because every test must set up all of its dependencies, and hard to change, because a change for one responsibility risks breaking another. A package can be treated as a bounded context with a single responsibility: the monorepo layout should mirror that, with shared types in one package, retrieval logic in another, and generation in a third. Keeping responsibilities at the package boundary is as important as keeping them at the class boundary.

## Open/Closed Principle

Software entities should be open for extension but closed for modification. You should be able to add new functionality without changing existing code. In Python this is achieved through abstract base classes (ABC), protocols, and dependency injection. The implementation strategy has two parts. Define abstract interfaces in a shared package and implement them in specific packages, then use a factory or registry pattern to select implementations at runtime. Adding a new implementation should not require changing existing code, only registering the new implementation. The registry is the extension point: new chunkers, new embedders, or new rerankers plug in by calling a registration function rather than by editing a conditional chain.

Long if/elif/else chains that select behavior based on a type string violate OCP, because adding a new type requires modifying the chain. The fix is to replace the chain with a registry mapping types to implementations. This keeps the dispatch table data-driven and makes it impossible to forget a branch when adding a new type, because the registry is the single place new types are wired in. In practice this pattern shows up wherever the codebase has a type discriminator: document types choose a chunker, embedding model names choose a client, and provider names choose a reranker. Each of these is a case where the type string should drive a lookup, not an if/else cascade.

## Liskov Substitution Principle

Subtypes must be substitutable for their base types. If S is a subtype of T, then objects of type T may be replaced with objects of type S without altering the desirable properties of the program. This is a contract between a base type and its implementations: callers written against the base type must work unchanged when handed any implementation. Subtypes must accept the same input types (or broader), return the same output types (or narrower), not strengthen preconditions, not weaken postconditions, and not remove exceptions that callers expect.

When a protocol or ABC is used as the boundary type, every implementation must honor the same contract. A concrete implementation that raises an error the protocol does not declare, or that narrows the accepted input types, breaks substitution and will fail at runtime in ways the type checker cannot always catch. In a typed codebase, protocols and ABCs are the contract surface, and mypy can verify signatures but not behavioural constraints. Substitution tests verify interchangeability: create test cases that use the parent type and run them against all implementations. These tests catch contract violations early. In a monorepo this means running the shared behaviour suite against every registered implementation, so that a new implementation cannot silently diverge from the contract its predecessors honored.

## Interface Segregation Principle

Clients should not be forced to depend on interfaces they do not use. Large, fat interfaces should be split into smaller, more specific ones. In Python, protocols are the natural tool for ISP: a protocol declares just the methods a client needs, and a class satisfies the protocol implicitly if it provides those methods. Instead of one large Repository interface with CRUD plus search plus export methods, define separate Reader, Writer, and Deletable protocols. A read-only service only depends on Reader, not on the full Repository.

This keeps the dependency surface small: a service that only reads does not need to know about write methods, and the implementation can change its write behaviour without affecting read-only callers. A fat interface forces every implementer to provide methods it does not need, and every client to be coupled to methods it does not call. Fat interfaces make mock objects awkward and make it hard to swap in a stub for tests, because the stub must implement methods that are irrelevant to the test. The practical signal of an ISP violation is a mock that raises NotImplementedError for methods it never uses, or an interface that grows a new method and forces every implementer to update. Splitting interfaces along client boundaries keeps each dependency narrow and each mock honest.

## Dependency Inversion Principle

High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions. The direction of dependency follows the direction of abstraction, not the direction of control flow. This inverts the naive dependency structure where business logic imports the database driver directly. Dependencies should be injected via constructors or function parameters rather than instantiated directly. In FastAPI, use Depends() for dependency injection. This makes the system testable because dependencies can be swapped with mocks.

A service that receives its repository as a constructor argument can be tested with an in-memory fake without a live database. Instantiating concrete classes inside business logic violates DIP: for example, creating a PostgresUserRepository() inside a service method couples the service to the database implementation. The fix is to accept the repository as a parameter or constructor argument. The same reasoning applies to external clients: business code should depend on a narrow protocol, and the concrete client that talks to a hosted API should be wired in only at the composition root. The dependency inversion principle is what makes the other four principles operational: it is the mechanism by which the domain logic stays independent of the infrastructure it uses, and it is the pattern that lets a monorepo swap implementations without editing the callers.

## Quick Reference Table

| Principle | Goal | Red Flag | Python Pattern |
|-----------|------|----------|----------------|
| SRP | One reason to change | Class does 3+ jobs | Separate into service/repository/validator |
| OCP | Extend without modifying | Adding if/else chains | Use ABC or Protocol for abstraction |
| LSP | Substitutable types | Subclass breaks parent contract | Ensure type compatibility |
| ISP | Focused contracts | Clients have unused methods | Split fat interfaces |
| DIP | Depend on abstractions | Concrete instantiation in functions | Inject via Depends() or constructor |
