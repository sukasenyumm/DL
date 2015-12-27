﻿using System;

namespace LibDL.Utils
{
    public sealed class ThreadSafeRandom : Random
    {

        private object _lock = new object();

        public override int Next()
        {

            lock (_lock) return base.Next();

        }

        public override int Next(int maxValue)
        {

            lock (_lock) return base.Next(maxValue);

        }

        public override int Next(int minValue, int maxValue)
        {

            lock (_lock) return base.Next(minValue, maxValue);

        }

        public override void NextBytes(byte[] buffer)
        {

            lock (_lock) base.NextBytes(buffer);

        }

        public override double NextDouble()
        {

            lock (_lock) return base.NextDouble();

        }

    }
}
