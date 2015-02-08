organization := "com.hyperdata.ltr"

name := "lambda-mart"

version := "1.0"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  "commons-io" % "commons-io" % "2.4",
  "org.apache.commons" % "commons-lang3" % "3.3.2",
  "org.apache.spark" %% "spark-core" % "1.2.0",
  "org.apache.spark" %% "spark-mllib" % "1.2.0",
  "org.json4s" %% "json4s-native" % "3.2.11"
)

resolvers ++= Seq(
  Resolver.mavenLocal,
  "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"
)

incOptions := incOptions.value.withNameHashing(true)